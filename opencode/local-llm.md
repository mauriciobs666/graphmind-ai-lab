# Creating an OpenCode Agent

A running tutorial on configuring custom agents in OpenCode.

## Project layout

We're building a per-project agent named **Severino** under `severino/`:

```
claude/
├── tutorial.md
└── severino/
    ├── README.md
    └── opencode.json   # agent config (project-scoped)
```

Project-scoped config beats global config when you run `opencode` from inside `severino/`.

## 1. Locate or create your config file

OpenCode looks for config in this order:

- `./opencode.json` (project-level)
- `~/.config/opencode/config.json` (global)

## 2. Define an agent

```json
{
  "agent": {
    "my-agent": {
      "description": "A short description of what this agent does",
      "mode": "primary",
      "model": "anthropic/claude-sonnet-4-7",
      "prompt": "You are a specialized assistant focused on..."
    }
  }
}
```

> **The JSON key (`my-agent`) is the agent's identifier.** Pass it to `--agent`. Do **not** set a `name` field — it conflicts with the lookup and you'll see "Agent not found" errors even though `agent list` shows it.

## 3. Key agent fields

| Field         | Required | Description                                                                                |
| ------------- | -------- | ------------------------------------------------------------------------------------------ |
| `model`       | yes      | Model ID in `<provider-key>/<model-id>` form. Splits on first `/`, so embedded slashes OK. |
| `mode`        | recommended | `primary` (selectable as the main agent) or `subagent` (only callable via task tool)    |
| `prompt`      | no       | Custom system prompt                                                                       |
| `description` | no       | Shown in agent selector                                                                    |
| `tools`       | no       | Map of tool-name → boolean. Set `false` to disable specific tools                          |

## 4. Use the agent

Launch OpenCode and switch agents with the `/agent` command, or start with:

```bash
opencode --agent my-agent
```

## Notes

- The top-level key is **`agent`** (singular). `agents` plural is from older OpenCode versions and now errors.
- Agents inherit default provider settings unless overridden.
- Multiple agents let you switch between e.g. a fast cheap agent and a powerful one.
- For the full schema, run `opencode --help` or check the OpenCode docs.

## Connecting to a local LM Studio

LM Studio ships an OpenAI-compatible server, so it plugs into OpenCode as a custom provider.

### Step 1 — Load a model in LM Studio with enough context

1. Load a model in LM Studio (e.g. `mistralai/ministral-3-3b`).
2. **Before clicking Load, set the Context Length to at least `16384`** (32K recommended). The default `4096` is too small — OpenCode's system prompt alone is several thousand tokens before any conversation. You'll otherwise see `n_keep >= n_ctx` errors.
3. Push GPU Offload up as far as VRAM allows.
4. Click **Load Model**.

### Step 2 — Start the local server

1. Open **Developer → Local Server**.
2. Click **Start Server**. Default endpoint is `http://localhost:1234/v1`.
3. Copy the exact model ID shown — you'll need it in the config.

### Step 3 — Add the provider in `opencode.json`

```json
{
  "provider": {
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio",
      "options": {
        "baseURL": "http://localhost:1234/v1",
        "apiKey": "lm-studio"
      },
      "models": {
        "mistralai/ministral-3-3b": {
          "name": "Ministral 3B (local)"
        }
      }
    }
  },
  "agent": {
    "severino": {
      "mode": "primary",
      "model": "lmstudio/mistralai/ministral-3-3b",
      "prompt": "You are Severino, a Brazilian lawyer and legal expert running locally via LM Studio."
    }
  }
}
```

### Step 4 — Key fields

| Field             | Purpose                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `npm`             | `@ai-sdk/openai-compatible` — works for any OpenAI-style API                                                  |
| `options.baseURL` | LM Studio server URL (must include `/v1`)                                                                    |
| `options.apiKey`  | Any non-empty string. LM Studio ignores it; the adapter still requires it.                                   |
| `models.<id>`     | Must match the model ID LM Studio reports, slashes and all                                                   |
| `model` (agent)   | Format: `<provider-key>/<model-id>`. Splits on the first `/`, so `lmstudio/mistralai/ministral-3-3b` is fine. |

### Step 5 — Run it

```bash
cd severino
opencode --agent severino
```

### Picking a model

Not every GGUF works equally well with OpenCode. Things to look for:

- **Supports the `system` role.** Some Mistral GGUFs reject it (`Only user and assistant roles are supported`). The `lmstudio-community` re-uploads usually fix this. Ministral 3B, Gemma 3, Qwen 3, Llama 3 all handle system roles correctly.
- **Non-reasoning, or supports disabling reasoning.** A reasoning model (Qwen3 with `/think`, DeepSeek R1, etc.) can take a minute to answer "hello" because most of its tokens are spent on hidden reasoning. For an interactive agent, prefer a plain instruct model.
- **3B–9B is usually plenty** for a personal agent. Bigger = slower + more VRAM.

### Troubleshooting

- **`Unrecognized key: agents`** → Use `agent` (singular), not `agents`.
- **`Agent not found: "X"`** → Don't set a `name` field on custom agents; use the JSON key directly with `--agent <key>`.
- **`API key required`** → Add `options.apiKey` to the provider, or `set OPENAI_API_KEY=lm-studio`.
- **`Connection refused`** → LM Studio's server isn't running.
- **`Model not found`** → The agent's `model` must be `<provider-key>/<model-id>`, and `<model-id>` must match LM Studio's reported ID exactly.
- **`n_keep: X >= n_ctx: 4096`** → Reload the model with a larger Context Length (16K+).
- **`Only user and assistant roles are supported`** → The model's chat template rejects `system` messages. Pick a different model or use the lmstudio-community re-upload.
- **Painfully slow responses** → Reasoning model. Either disable thinking with `/no_think` (works on official Qwen3) or switch to a plain instruct model like Ministral.
- **Slow first response only** → LM Studio is JIT-loading the model. Eject and explicitly pre-load to avoid the timeout.
