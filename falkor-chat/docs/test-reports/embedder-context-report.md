# Embedder context — over-context behavior and per-model load context

> **Status:** active · **Owner:** `devops` · **Tracks:** K-099

## Probe: over-context embed

**Question (K-099 remaining item 1):** when `/v1/embeddings` gets an input longer than the
embedder's loaded context, does LM Studio reject it or silently truncate it?

**Builds observed on.** LM Studio app `0.4.26+4` (`resources/app/package.json`), `lms` CLI
commit `dcefd5a`, selected GGUF engine `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.67.0`
(`lms runtime ls`, ✓ column). These were read read-only after the probe. The 2.67.0 engine
directories were installed 2026-10-30 21:32 local, before it.

**Setup (2026-10-31, ~23:33 UTC).** LM Studio at `http://localhost:1234` (the `lmstudio` baseURL in
`~/.config/opencode/opencode.local.json`, resolved the way `scripts/shlib/opencode_config.sh` does;
`FALKORCHAT_OPENCODE_CONFIG` unset). A dedicated probe instance, loaded by this run and unloaded
by it afterwards, so no shared instance was used:

```
lms.exe load text-embedding-qwen3-embedding-0.6b -c 512 --gpu max --identifier k099-probe-embedder-512 -y
```

`lms ps --json` (the authority — `GET /api/v0/models` reported `loaded_context_length: 2048` for
this same instance, the known misreport):

```
{"identifier": "k099-probe-embedder-512", "modelKey": "text-embedding-qwen3-embedding-0.6b", "type": "embedding", "contextLength": 512, "maxContextLength": 32768, "status": "idle"}
```

**Input.** Non-repeating filler, `" ".join(f"item{i} costs {i % 97} units" for i in range(n))`,
so every prefix length gives a distinct text. `POST /v1/embeddings {"model": <identifier>, "input": <text>}`.

**Raw output, run 1** (scratch script `probe_embed_ctx.py`; its token calibration failed because
LM Studio returns `usage.prompt_tokens: 0` for embeddings, so the "~2x ctx" labels are item
counts from a 5-tokens/word guess, not measured tokens — the chars are exact):

```
2026-10-31T23:34:40Z
--- control (200 words): chars=4463 HTTP 200 in 0.29s
    usage={'prompt_tokens': 0, 'total_tokens': 0} dim=1024 first3=[-0.054611023515462875, 0.01839401386678219, -0.004022330977022648]
    calibration: 0 prompt tokens / 200 words = 5.00 tokens/word
--- over-context (~2x ctx, 204 words): chars=4551 HTTP 200 in 0.05s
    usage={'prompt_tokens': 0, 'total_tokens': 0} dim=1024 first3=[-0.054611023515462875, 0.01839401386678219, -0.004022330977022648]
--- prefix (~0.9x ctx, first 92 words): chars=2003 HTTP 200 in 0.04s
    usage={'prompt_tokens': 0, 'total_tokens': 0} dim=1024 first3=[-0.054611023515462875, 0.01839401386678219, -0.004022330977022648]
--- suffix (~0.9x ctx, last 92 words): chars=2105 HTTP 200 in 0.03s
    usage={'prompt_tokens': 0, 'total_tokens': 0} dim=1024 first3=[0.0329539030790329, -0.02172214165329933, -0.0029047192074358463]
--- far over (~8x ctx, 819 words): chars=18636 HTTP 200 in 0.04s
    usage={'prompt_tokens': 0, 'total_tokens': 0} dim=1024 first3=[-0.054611023515462875, 0.01839401386678219, -0.004022330977022648]
cos(long, prefix) = 1.000000
cos(long, suffix) = 0.460863
cos(prefix, suffix) = 0.460863  (baseline: two different texts)
```

Every call answered **HTTP 200** with a normal 1024-dim vector — up to 18,636 chars against a
512-token context. The 2,003-, 4,463-, 4,551- and 18,636-char inputs all returned the **identical**
vector (cos 1.000000): everything past the head is dropped. The last 92 items, which share no
text with the head, return a different vector (cos 0.46).

**Raw output, run 2 — where the cut falls** (scratch script `probe_cutoff.py`: binary search for
the smallest prefix whose vector equals the 2,000-item text's):

```
{"model": "k099-probe-embedder-512", "max_items": 2000, "full_chars": 46679, "cutoff_items": 67, "cutoff_prefix_chars": 1453, "last_differing_items": 66, "last_differing_chars": 1431, "cutoff_tail": "4 costs 64 units item65 costs 65 units item66 costs 66 units"}
```

The vector stops changing after 67 items / 1,453 chars ≈ 512 tokens at ~7.6 tokens per item
(` item`, two single digits, ` costs`, two digits, ` units` — Qwen tokenizes digits singly). No
tokenizer is installed to count exactly; run 3 checks the cut scales with the loaded context.

**Raw output, run 3 — the same search at the shipping context** (probe instance
`k099-probe-embedder-4096`, `-c 4096`, `lms ps --json` `"contextLength": 4096`; 2026-10-31T23:35:42Z):

```
{"model": "k099-probe-embedder-4096", "max_items": 4000, "full_chars": 94469, "cutoff_items": 473, "cutoff_prefix_chars": 10718, "last_differing_items": 472, "last_differing_chars": 10695, "cutoff_tail": "costs 82 units item471 costs 83 units item472 costs 84 units"}
```

473 items ≈ 10 one-digit + 90 two-digit (~7.6 tokens) + 373 three-digit (~8.6 tokens) items
≈ 4,000 tokens: the cut scales with the loaded context, 512 → ~67 items, 4096 → ~473.

The LM Studio server log (`%USERPROFILE%\.lmstudio\server-logs\2026-10\2026-10-31.1.log`) shows only
`Received request to embed` / `Returning embeddings (not shown in logs)` for every probe call — no
truncation warning, no error.

### Verdict

**On the builds above: silent head truncation, not a rejection.** An over-context `/v1/embeddings` input returns
HTTP 200 and a normal vector computed from the first *loaded-context* tokens; the rest is dropped
without a status, a body field (`usage` is all zeros) or a log line. So for falkor-chat:

- An over-long message is **not** dropped from search (K-099's feared "rejection fails the embed
  job silently" does not happen); it is findable only by its first ~4,096 tokens. At ~4 chars per
  token for prose, that is ~16,000 chars — every REST message (`MAX_TEXT_LEN` = 8,000 chars) and
  every document chunk fits whole at 4096. Dense text (digits, code) runs ~2.3–2.8 chars per token:
  the probe's filler cut at 10,718 chars, so an 8,000-char message of that kind still fits.
- The MCP `send_message` gap (item 3) therefore degrades recall for an over-long agent post's tail,
  rather than losing the post; capping it at `MAX_TEXT_LEN` keeps it inside the 4096 window.
- Nothing in the response tells a caller it was truncated: a length guard has to be enforced
  before the call, not detected after it.
- **The verdict is build-specific.** Upstream llama.cpp's server *rejects* an embed input longer
  than `n_ubatch` ("input is too large to process"; see the review
  `falkor-chat/docs/reviews/embedder-context.md`, P2-3). An LM Studio app or runtime update can
  therefore turn truncation into rejection, and with it K-099's original failure mode (embed job
  fails, log line only, message missing from vector search). Re-run this probe after such an
  update and compare against the versions above.

### Incidental observation — `laya` left the loaded set during the probe

At 20:33:04 local (`lms ps --json`), the only loaded model was `laya` (the `decision`-type model
model-bench's typed-decision-models work uses; `"ttlMs": null`, `parallel: 4`, context 8192). At
20:33:51, right after this run's `lms load … --identifier k099-probe-embedder-512` (log: 20:33:43),
`lms ps --json` listed only the probe instance. The server log has no unload line for `laya`
(a CLI `unloadModel` *is* logged — this run's own unloads are), and LM Studio's settings carry
`developer.unloadPreviousJITModelOnLoad: true`. Whether this run's load evicted `laya` (auto-evict
treating it as a JIT-loaded model) or a concurrent agent unloaded it over REST is **not
determined**; this run did not reload it, because its load config (`physical_batch_size` etc.)
belongs to model-bench's harness.

**Process note:** a live probe on a shared LM Studio should record `lms ps --json` immediately
before and after each of its own loads, and say up front that a load can evict JIT-loaded models.

## Change: the embedder loads at its own context

`scripts/ensure_demo_models.py` gains `EMBED_CONTEXT_LENGTH = 4096` beside
`DEMO_CONTEXT_LENGTH = 8192`; `DemoSet.context_for(id)` picks the embedder's own context, else the
chat pool's, and `load_args(id, ctx)` / the load-it-yourself instructions use it per model.

**Already-loaded embedder: checked, advisory only.** On every exit-0 path the script reads
`lms ps --json` (the API misreports embedder context) and prints a `WARNING` with the reload command
when the embedder's identifier is not at 4096; exit stays 0 and nothing is reloaded. A warning
rather than a refusal because the probe above shows a mismatch cannot break the demo: larger only
costs VRAM, and smaller embeds only a long input's head on the builds probed above (the
storefront itself never embeds; the docstring and warning cite this report rather than state it).
A refusal would push the operator to unload a model the `:8200` agent-team server embeds through.
The chat-model refusal on a non-8192 context is unchanged. With no CLI, a bad `LMS_CLI`, an
unreadable `lms ps`, not exactly one instance under the embedder's identifier, or a non-integer
`contextLength`, the script prints a "not checked" note, so the no-op path still needs no CLI.
The exit-0 wording claims only what was checked: "chat model at the demo context — nothing to
load (embedder context checked below)".

**Tests** — `server/tests/test_ensure_demo_models_script.py` (the stub `lms` CLI now answers
`ps --json` with each instance's real context): **53 passed**; `test_start_demo_script.py` +
`test_opencode_config_resolution.py` 32 passed. Mutation table (each mutant applied to a copy-
restored `ensure_demo_models.py`, suite run with `-x`; original restored byte-identical). MA–MG
are review Pass 2's mutants (`falkor-chat/docs/reviews/embedder-context.md`):

| # | Mutant | Result |
|---|---|---|
| M1 | embedder falls back to 8192 (`context_for` → `DEMO_CONTEXT_LENGTH`) | killed |
| M2 | every model at 4096 | killed |
| M3 | contexts swapped | killed |
| M4 | `lms load` ignores the per-model context | killed |
| M5 | load-it-yourself hint ignores the per-model context | killed |
| M6 | `lms ps` check never warns | killed |
| M7 | `lms ps` mismatch runs `lms unload` | killed |
| M8 | `lms ps` mismatch refuses | killed |
| M9 | `lms ps` check skipped on the no-op path | killed |
| M10 | `EMBED_CONTEXT_LENGTH = 8192` | killed |
| MA | load-it-yourself command at 4096 for every model | killed |
| MB | LM Studio app hint at 4096 for every model | killed |
| MC | bad `LMS_CLI` on the no-op path left uncaught (exit 1) | killed |
| MD | `lms ps` exit code ignored | survives — accepted: a failed `ps` with valid JSON on stdout is implausible (the review's call too) |
| ME | 0 or 2 embedder instances not treated as "not checked" | killed |
| MF | embedder check dropped after a load | killed |
| MG | non-integer `contextLength` guard dropped | killed |

**Live run** (2026-10-31, nothing loaded beforehand, so additive):

```
ensure_demo_models: not loaded: mistralai/ministral-3-3b, text-embedding-qwen3-embedding-0.6b
ensure_demo_models: loading (can take a minute): …/lms.exe load mistralai/ministral-3-3b -c 8192 --gpu max --identifier mistralai/ministral-3-3b -y
ensure_demo_models: loading (can take a minute): …/lms.exe load text-embedding-qwen3-embedding-0.6b -c 4096 --gpu max --identifier text-embedding-qwen3-embedding-0.6b -y
ensure_demo_models: loaded: mistralai/ministral-3-3b, text-embedding-qwen3-embedding-0.6b — demo models ready
ensure_demo_models: embedder context 4096 (lms ps)
exit=0
```

`lms ps --json` afterwards (fields trimmed):

```
{"identifier": "mistralai/ministral-3-3b", "modelKey": "mistralai/ministral-3-3b", "type": "llm", "contextLength": 8192, "maxContextLength": 262144, "status": "idle"}
{"identifier": "text-embedding-qwen3-embedding-0.6b", "modelKey": "text-embedding-qwen3-embedding-0.6b", "type": "embedding", "contextLength": 4096, "maxContextLength": 32768, "status": "idle"}
```

A second run printed `nothing to do` and `embedder context 4096 (lms ps)`, exit 0. The VRAM and
paging re-measure (K-099 item 4) is not part of this report.
