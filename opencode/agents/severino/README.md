# Severino

A local OpenCode agent powered by [LM Studio](https://lmstudio.ai/). Currently configured as a **read-only coding advisor** — reads, reviews, explains, and debugs code, proposing changes as snippets/diffs for you to apply (he can't modify files himself). See [Customizing Severino](#customizing-severino) to repurpose him.

## Prerequisites

- [OpenCode](https://opencode.ai/) installed (`npm install -g opencode-ai`) and on your `PATH`
- [LM Studio](https://lmstudio.ai/) installed with at least one chat-capable model downloaded

## Setup

### 1. Load a model in LM Studio with enough context

1. Open LM Studio.
2. Pick a model (see [Choosing a model](#choosing-a-model) below). Default for this project is `nvidia/nemotron-3-nano-4b`.
3. **Before clicking Load, set the Context Length to at least `16384`** (32K recommended). The default 4096 is too small — OpenCode's system prompt alone is several thousand tokens before any conversation, and you'll hit `n_keep >= n_ctx` errors.
4. Push GPU Offload up as far as your VRAM allows.
5. Click **Load Model** and wait for the green "loaded" indicator.

### 2. Start the LM Studio server

1. Go to **Developer → Local Server**.
2. Click **Start Server**. Default endpoint is `http://localhost:1234/v1`.
3. Confirm the model is listed by hitting `curl http://localhost:1234/v1/models`.

### 3. Match the model ID in `opencode.json`

Edit [`opencode.json`](opencode.json):

- Under `provider.lmstudio.models`, the key **must match the LM Studio model ID exactly** (slashes included).
- The agent's `model` field is in the form `lmstudio/<model-id>`. OpenCode splits on the first `/`, so `lmstudio/nvidia/nemotron-3-nano-4b` resolves to provider `lmstudio` + model `nvidia/nemotron-3-nano-4b`.

### 4. Run Severino

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

LM Studio runs any GGUF, but not every model is OpenCode-friendly:

| Trait                    | Why it matters                                                                                                                                                                                                                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Handles `system` role**| OpenCode sends a system prompt. Some Mistral GGUFs (e.g. vanilla `mistral-7b-instruct-v0.3`) reject it — Jinja errors with `Only user and assistant roles are supported`. Prefer `lmstudio-community` re-uploads, or models like Ministral, Gemma 3, Qwen 3, Llama 3 that handle it natively.       |
| **Non-reasoning**        | Reasoning models (Qwen3 with `/think`, DeepSeek R1, etc.) spend most tokens on hidden reasoning before answering. Great for hard problems, painful for casual chat (10× slower). For an interactive agent, prefer plain instruct models or ones with reliable `/no_think` support.                  |
| **Modest size**          | 3B–9B is usually plenty for a personal assistant. Bigger = slower and more VRAM.                                                                                                                                                                                                                    |

Default for this project: **`nvidia/nemotron-3-nano-4b`** — small, fast, handles `system` messages, and in practice noticeably better than Ministral 3B at following the persona prompt and reasoning about code coherently. Ministral 3B (`mistralai/ministral-3-3b`) is kept in `opencode.json` as an alternate; swap by changing the agent's `model` field.

## Troubleshooting

| Symptom                                                       | Cause / Fix                                                                                                                                                                                                |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Unrecognized key: agents`                                    | The top-level key must be **`agent`** (singular), not `agents`.                                                                                                                                            |
| `Agent not found: "Severino"` despite being in the list       | A `name` field on the agent shadows the JSON key. Remove `name` from the agent definition; use the JSON key (`severino`) as the identifier.                                                                |
| `API key required`                                            | Set `provider.lmstudio.options.apiKey` to any non-empty string (e.g. `"lm-studio"`), or export `OPENAI_API_KEY=lm-studio`.                                                                                  |
| `Connection refused`                                          | LM Studio's local server isn't running. Start it from **Developer → Local Server**.                                                                                                                        |
| `Model not found`                                             | The agent's `model` value must be `<provider-key>/<model-id>`, and `<model-id>` must match LM Studio's reported ID exactly (slashes included).                                                              |
| `n_keep: X >= n_ctx: 4096`                                    | Model was loaded with too-small context. **Eject** the model in LM Studio, reload with **Context Length** set to 16K+ (32K recommended).                                                                    |
| `Only user and assistant roles are supported`                 | The model's chat template rejects `system` messages. Switch to an `lmstudio-community` re-upload, pick a different model (Ministral, Gemma 3, Qwen 3), or override the prompt template in LM Studio.       |
| Response takes 1+ minute on a short prompt                    | Reasoning model. Append `/no_think` to the system prompt (works on official Qwen3, unreliable on community fine-tunes) or switch to a non-reasoning model.                                                  |
| First response very slow even after warmup                    | LM Studio is JIT-loading the model. Eject and explicitly reload beforehand to avoid the timeout.                                                                                                            |
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
