# Severino

A local OpenCode agent powered by [LM Studio](https://lmstudio.ai/). Currently configured as a **read-only coding advisor** — reads, reviews, explains, and debugs code, proposing changes as snippets/diffs for you to apply (he can't modify files himself). See [Customizing Severino](#customizing-severino) to repurpose him.

## Prerequisites

- [OpenCode](https://opencode.ai/) installed (`npm install -g opencode-ai`) and on your `PATH`
- [LM Studio](https://lmstudio.ai/) installed with at least one chat-capable model downloaded

## Setup

### 1. Load a model and start the LM Studio server

Follow the general walkthrough in [`opencode/local-llm.md`](../../local-llm.md)
(§ *Connecting to a local LM Studio*, steps 1–2): load a model **with Context
Length ≥ 16384** (32K recommended), then start the local server at
`http://localhost:1234/v1`. Severino-specific deltas:

- Default model for this project is **`nvidia/nemotron-3-nano-4b`** (see [Choosing a model](#choosing-a-model)).
- Confirm the model is listed with `curl http://localhost:1234/v1/models`.

### 2. Match the model ID in `opencode.json`

Edit [`opencode.json`](opencode.json):

- Under `provider.lmstudio.models`, the key **must match the LM Studio model ID exactly** (slashes included).
- The agent's `model` field is in the form `lmstudio/<model-id>`. OpenCode splits on the first `/`, so `lmstudio/nvidia/nemotron-3-nano-4b` resolves to provider `lmstudio` + model `nvidia/nemotron-3-nano-4b`.

### 3. Run Severino

From this directory:

```bash
opencode --agent severino
```

Or launch OpenCode and switch with `/agent severino`.

## Configuration reference

See [`opencode.json`](opencode.json). Key pieces:

| Field                                  | Purpose                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| `provider.lmstudio.npm`                | `@ai-sdk/openai-compatible` — adapter for any OpenAI-shaped HTTP server              |
| `provider.lmstudio.options.baseURL`    | LM Studio server URL (must include `/v1`)                                            |
| `provider.lmstudio.options.apiKey`     | Dummy string (e.g. `lm-studio`). The adapter requires it; LM Studio ignores it.      |
| `provider.lmstudio.models`             | Map of model IDs LM Studio reports                                                   |
| `agent.severino.mode`                  | `primary` (selectable as main agent) or `subagent` (only invokable via the task tool)|
| `agent.severino.model`                 | `lmstudio/<model-id>`                                                                |
| `agent.severino.prompt`                | System prompt shaping Severino's behavior                                            |
| `agent.severino.tools`                 | Map of tool-name → boolean. Set `false` to disable specific tools                    |

> **Two non-obvious things:**
> - The top-level key is **`agent`** (singular), not `agents`.
> - Do **not** set a `name` field on a custom agent. The JSON key (`severino`) is the identifier `--agent` looks up; adding `name` shadows the key and breaks the lookup.

## Choosing a model

What makes a GGUF OpenCode-friendly (handles the `system` role, non-reasoning, 3B–9B size) is covered in [`opencode/local-llm.md`](../../local-llm.md) § *Picking a model*.

Default for this project: **`nvidia/nemotron-3-nano-4b`** — small, fast, handles `system` messages, and in practice noticeably better than Ministral 3B at following the persona prompt and reasoning about code coherently. Ministral 3B (`mistralai/ministral-3-3b`) is kept in `opencode.json` as an alternate; swap by changing the agent's `model` field.

## Troubleshooting

The general LM-Studio-with-OpenCode symptoms (`Unrecognized key: agents`, `Agent not found`, `API key required`, `Connection refused`, `Model not found`, `n_keep >= n_ctx`, `Only user and assistant roles are supported`, slow reasoning models, slow JIT-loaded first response) are covered in [`opencode/local-llm.md`](../../local-llm.md) § *Troubleshooting*. Severino-specific:

| Symptom                                                       | Cause / Fix                                                                                                                                                                                                |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TUI launches but agent picker doesn't show Severino           | You didn't `cd` into `severino/` before running `opencode`. Project-scoped config is only loaded from the current working directory.                                                                        |

## Customizing Severino

Severino's persona lives entirely in `agent.severino.prompt` in [`opencode.json`](opencode.json) — he's currently a read-only coding advisor, but the same agent can be repurposed by rewriting that one string. Examples of alternate personas:

- *"You are Severino, a Brazilian law expert..."*
- *"You are Severino, an English-to-Portuguese translator..."*
- *"You are Severino, a Linux sysadmin who answers in concise shell snippets..."*

Restart OpenCode after changing the prompt.

### Enabling tools

By default Severino has `read`, `glob`, and `grep` enabled (so he can look at files when asked) and everything else off. To let him run bash, edit files, fetch web pages, etc., flip the relevant entries in `agent.severino.tools` to `true`. Each enabled tool adds 500–1500 tokens of schema to the system prompt — bump LM Studio's Context Length if you enable many.

> **Note for the coding-advisor persona:** `read`/`glob`/`grep` are deliberately kept on so Severino can inspect the project, while `bash`/`edit`/`write` stay off — he proposes changes as snippets/diffs rather than applying them. If you want him to actually edit files or run commands, flip those entries to `true` (and bump LM Studio's Context Length accordingly), but note a 4B local model can mangle files or run shell commands unreliably.
