# LM Studio / local small-model realism — this lab's stack

> **Live-verified knowledge base for `data-scientist`.** Facts confirmed by direct testing against
> this lab's LM Studio server, not assumed from model cards. Model behavior across versions/quants
> is perishable — treat entries as **verified for the cited model tag and date**, re-check before
> leaning on one for a live decision.
>
> **This is a cache, not the source of truth.** Origin: distilled 2026-08-11 from the
> `data-scientist` agent's learnings inbox via `agent-maintenance` skill §5.

## On a terminal tool-call schema, Ministral-3B was MORE reliable than Qwen3-4B — native `tool_calls` vs. prose

Direct replay of the answer-node `post_message` schema against LM Studio (`:1234`):
`mistralai_ministral-3-3b-instruct-2512` emitted a native OpenAI `tool_calls` `post_message` 3/3
draws (parsed cleanly by this lab's `llm.py`); `qwen/qwen3-4b-2507` emitted plain prose with **no**
tool call 3/3 draws. LM Studio's OpenAI-compat layer surfaced Ministral's tool call correctly on
this path — a documented risk that "Mistral's tool-call format won't parse" did **not** materialize
on the native path.

**Consequence:** the naive prior "smaller parameter count ⇒ worse at structured tool calls" did not
hold for this specific pair on this specific schema — verify per model/schema rather than ranking
models by size alone for tool-calling reliability. (Ministral never reached the live answer node in
the coordination this was observed in, so this is a capability-probe datapoint, not a banked
production result — re-verify before treating it as decided.)

**Context:** K-022 D13 Qwen-vs-Ministral capability probe (falkor-chat live-triage reliability
work), classifying Defect-C / D4 (genuine no-post).

## Mistral/Ministral GGUF chat templates enforce strict user/assistant role alternation — HTTP 400 on two consecutive same-role messages; Qwen3 tolerates it silently

Live curl/urllib against `localhost:1234` `v1/chat/completions`: `[system, user, user]` → HTTP 400
(`"conversation roles must alternate..."`) on both `mistralai_ministral-3-3b-instruct-2512` and
`mistralai/ministral-3-3b` catalog ids; the identical message shape against `qwen/qwen3-4b-2507` →
200 OK. Reproduced against `falkor-chat`'s real `triage@v1` intake node prompt (system + user
trigger + a trailing user-role CONTEXT block).

**Consequence:** any code that unconditionally appends a trailing same-role block after arbitrary
prior turns (e.g. `falkor-chat`'s `executor._assemble_messages` appending a final user-role CONTEXT
block after thread turns) hard-crashes the first time the thread ends on that same role — on a
Mistral-family model only; it works fine on Qwen. Check role alternation explicitly before
assuming a prompt-assembly pattern that works on one model family ports to another.

**Context:** `falkor-chat` K-027 item 5 Ministral re-probe (`docs/plans/ministral-reprobe-ml.md`).

## LM Studio can expose two catalog ids for the same underlying weights — verify state-flipping or byte-identical completions before assuming two entries are two different models

On this lab's box, `mistralai_ministral-3-3b-instruct-2512` (publisher `bartowski`) and
`mistralai/ministral-3-3b` (publisher `mistralai`) alias to **one** loaded model slot: calling one
flips `/api/v0/models` state to `loaded` for it and `not-loaded` for the other, and
temperature=0 completions are byte-identical across both ids.

**Consequence:** don't assume two differently-named LM Studio catalog entries are two different
weight files without checking `/api/v0/models` state-flipping or a byte-identical-completion probe
first — a `curl :1234/v1/chat/completions` and `curl :1234/api/v0/models` round trip against both
ids is cheap and conclusive.

**Context:** `falkor-chat` K-027 item 5 Ministral re-probe, step 1.

## A live-run report's provenance (model/quant/temperature/baseURL) can silently diverge from the repo's static config, per box — verify live before trusting it

`falkor-chat`'s model resolution is two hand-edited files, and the provider file
(`FALKORCHAT_OPENCODE_CONFIG`, defaulting to `$HOME/.config/opencode/opencode.json`) is
**machine-local, outside the repo** — not something `git blame`/`grep` can verify. On one box that
default file declared `lmstudio` at an unreachable LAN IP listing only an unrelated model, while
`config/models.json`'s `defaults.guard` named `lmstudio/qwen/qwen3-4b-2507` — a model the provider
file never mentioned. `ProviderCatalog`/`_resolve_element` (`modelconfig.py`) validates only the
**provider id**, not the model id, so this kind of mismatch resolves silently (wrong/unreachable
`baseURL`) rather than failing loudly. Separately, the repo sets **no** `temperature` key anywhere
for any kind — an uncontrolled sampling parameter for any determinism-sensitive eval design that
assumes one is pinned near 0.

**Consequence — two reusable habits:** (1) before trusting a report's provenance header on any
project using a machine-local provider config, live-check the actually-reachable endpoint —
LM Studio's `curl :1234/api/v0/models` gives `quantization` and `state: loaded|not-loaded` per
model, exactly what a provenance header needs — rather than reading only the repo's static config,
since the two can diverge per-box with no loud failure. (2) grep the whole repo for `temperature`
(or the sampling-param equivalent) before writing any non-determinism-handling section (k
replicates, flip-rate, etc.) that assumes a pinned value.

**Context:** `falkor-chat/docs/plans/guard-judge-calibration-ml.md` (K-027 item 3).
