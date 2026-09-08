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
`baseURL`) rather than failing loudly. Separately, **what is pinned changes between
revisions, in both directions** — when this section was first written the repo set no
`temperature` key at all for any kind; `config/models.json`'s `models` block now pins
`temperature: 0` for `lmstudio/qwen/qwen3-4b-2507` and `lmstudio/mistralai/ministral-3-3b`.
Read it, never assume it.

**Consequence — two reusable habits:** (1) before trusting a report's provenance header on any
project using a machine-local provider config, live-check the actually-reachable endpoint —
LM Studio's `curl :1234/api/v0/models` gives `quantization` and `state: loaded|not-loaded` per
model, exactly what a provenance header needs — rather than reading only the repo's static config,
since the two can diverge per-box with no loud failure. (2) grep the whole repo for `temperature`
(or the sampling-param equivalent) before writing any non-determinism-handling section (k
replicates, flip-rate, etc.) that assumes a pinned value — **and never read a pin as
determinism.** A pinned `temperature: 0` on this stack is not run-to-run stable: 40
identical-script repeated conversations against `qwen/qwen3-4b-2507` put a behavioural
onset turn at 4 in 39/40 runs and at 3 in 1/40 — same prompt, same params, different
outcome — and `falkor-chat`'s later K-062 measurements found the pin did not narrow a much
wider swing between sessions either (`falkor-chat/docs/BACKLOG.md`). A pin buys
comparability, not repeatability: every eval on this stack is a sampling design, so report
a rate with its interval and never a single draw as a measurement.

**Context:** `falkor-chat/docs/plans/guard-judge-calibration-ml.md` (K-027 item 3).

## The machine-readable measurement surface is on `/api/v0/`, not `/v1/` — and `lms` is reachable from WSL only as `lms.exe`

**Verified 2026-09-07 on this box** (re-derived; originally observed 2026-09-02, timings and the
JIT clause added 2026-09-07 from a 2026-09-03 observation).

- **`POST /api/v0/chat/completions` carries the per-call measurement fields the OpenAI-compatible
  `/v1/chat/completions` route omits:** `stats{time_to_first_token, tokens_per_second,
  generation_time, stop_reason}`, `model_info{arch, quant, format, context_length}` and
  `runtime{name, version}` (LM Studio REST docs, `lmstudio.ai/docs/developer/rest/endpoints`).
  Anything that needs latency or a runtime fingerprint per call must use the `v0` route —
  and **only** that route: `POST /api/v0/embeddings` carries none of the three (verified
  2026-09-08 against `localhost:1234`, `text-embedding-qwen3-embedding-0.6b`; the response
  holds exactly `data`/`model`/`object`/`usage`, with `usage` itself
  `{prompt_tokens: 0, total_tokens: 0}`). So an **embeddings-only arm can never populate a
  `runtime`, `stats` or `model_info` field**, by any call it makes. "The v0 API" is not one
  uniform fingerprint source; the catalog route and the chat route are, and nothing else is.
- **`GET /api/v0/models` fingerprints the catalog; `GET /v1/models` cannot.** Live response on this
  box returns, per model, `id`, `object`, `type`, `publisher`, `arch`, `compatibility_type`,
  `quantization`, `state` (`loaded`/`not-loaded`), `max_context_length` and `capabilities`. The
  `/v1/` route returns only `id`/`object`/`owned_by`. Both routes are effectively free — 19 models
  came back in 1.6-6.5 ms over six calls, `/v1/` no faster than `/api/v0/` — so the choice between
  them is about content, never cost; poll either as often as you like.
- **Of those fields, `capabilities` is not a tool-calling gate — it carries no discriminating
  information at all.** Re-derived 2026-09-08 on this box (`curl -s :1234/api/v0/models`;
  originally observed 2026-09-02): every entry that has the key holds exactly
  `["tool_use"]` and **no entry holds anything else**, so the field never says *no*. It says
  `["tool_use"]` for the **embeddings** model `text-embedding-qwen3-embedding-0.6b`, and it is
  **absent entirely** from four entries spanning both kinds — two `vlm` and one `llm` chat model
  (`google/gemma-3-4b`, `google/gemma-3-12b`, `gemma-3-4b-vl-it-…`) plus a second embeddings model
  (`text-embedding-nomic-embed-text-v1.5`). So presence does not imply a chat model, and absence
  implies nothing at all. **Gate on `type` ∈ {`llm`, `vlm`}**, and treat `capabilities` as at best
  a non-blocking hint — a harness that refuses a tool-caller run "because the catalog lacks
  `tool_use`" will refuse four working models on this box today and admit an embedder.
  `loaded_context_length` is the same trap one field over: absent from **every** entry while
  `state == not-loaded` — read `max_context_length` instead, or load the model first. Confirmed
  from both sides on a third measurement (2026-09-08, later the same day): 15 of 16 entries
  `not-loaded` and keyless, while the one model a probe had just JIT-loaded carried the key.
  **The catalog's size is not stable and must never be hardcoded** — 19 models at the two earlier
  measurements, **16** at this one (10 `vlm`, 4 `llm`, 2 `embeddings`), so the `capabilities`
  census reads 12 of 16 rather than 15 of 19. What *is* stable across that turnover is the shape
  of the finding: the **same four** entries named above are still the ones with no `capabilities`
  key, and every entry that has it still holds only `["tool_use"]`. Recount before quoting a
  denominator; the conclusion survives without one.
- **The `lms` CLI is not on the WSL `PATH`** (`command -v lms` exits 1), but the Windows binary is
  reachable and works from WSL at `/mnt/c/Users/<user>/.lmstudio/bin/lms.exe`. Confirmed working
  this way: `lms server status --json` (→ `{"running":true,"port":1234}`), `lms ps --json` (→ `[]`
  with nothing loaded), and `lms load --estimate-only` (documented in `lms load --help` as
  "Calculate an estimate of the resources required to load the model. Does not load the model.").
  **Each `lms.exe` call costs ~0.30 s** (0.30/0.31/0.32 s over three `lms ps --json` runs) — the
  WSL-to-Windows subprocess price, not the command's own work. A fingerprint collector that shells
  out per field pays it per field; batch what you can, and prefer the HTTP routes above, which are
  ~100x cheaper, for anything they can answer.
- **Two gaps to design around.** `lms version` prints only a CLI commit hash (`CLI commit:
  <sha>`) — there is no LM Studio *app* version anywhere in the CLI output. And **no API field and
  no `lms load` flag exposes the KV-cache setting**; `lms load` offers `--context-length` but
  nothing for KV-cache quantization. Both must be operator-attested rather than machine-collected.
- **A request naming an unloaded model triggers LM Studio's JIT auto-load, so the first call pays
  the load.** `lms ps --json` returning `[]` does not mean a subsequent completion will refuse — it
  means the next one will be cold. **That cold load can also *fail* rather than merely be slow** —
  observed 2026-08-26 on an embedding model (`text-embedding-qwen3-embedding-0.6b`): the first
  request after an idle period returned HTTP 400 `Failed to load model … Error loading model`, a
  manual `curl` to `/v1/embeddings` against the same model then succeeded, and re-running the exact
  same call immediately after passed with no other change. A live test that hits this on its **first**
  call is an environment flake, not a code or config defect — retry once before diagnosing; a
  harness that must not flake should warm the model with a throwaway call first.
  **Do not size a design against any single measured load cost:**
  three cold loads measured on this box span ~3.5 s to ~21 s (≈6x). What is stable, and what a
  latency design should key on, is that LM-Studio-side `ttft` and `generation_time` **exclude** the
  JIT load while the client wall clock includes it — so **`wall − (ttft + generation_time)` is the
  load, isolated, and is therefore an in-call reload detector**: a reload that begins and ends
  inside one timed call is invisible to a between-item residency probe but shows up in that gap.
  The separation is not marginal — re-measured live on this box against `qwen/qwen3-4b-2507`
  (Q4_K_M) from a residency-confirmed `[]` start: cold gap **3 507 ms**, against **−6.5 … +6.8 ms**
  across 20 warm calls, ~500x. Threshold the gap's *magnitude*, never `gap > 0`: the warm gaps go
  negative because the client clock and the server's timers bracket different work. Full treatment,
  including the threshold's basis, in `docs/plans/small-model-benchmarking-ml.md` §11.4/§11.5.1.
  **The detector withholds on a covariate, not on latency**, so it is *not* right-censoring — a
  withheld call can be faster than a timed one (a 1.1 s gap around 0.2 s of generation is withheld
  at 1.3 s while a clean 2.0 s call beside it is timed). Any "every withheld call was slower than
  every timed call" claim has to be **computed per render**, never argued from the rule's shape.

**Consequence:** an environment fingerprint or latency report can be collected automatically from
`/api/v0/` plus `lms.exe`, except the app version and the KV-cache setting, which have to be
recorded by hand.

**Context:** `docs/plans/small-model-benchmarking.md` (model-bench FR-7 environment fingerprint,
FR-11 latency/RAM reporting).

## More targeted wording guidance is not monotonically safer — on `mistralai/ministral-3-3b`, a second iteration cut net correctness 80% → 70% by suppressing a protective multi-call self-correction

**Verified 2026-09-07** against `falkor-chat/docs/HISTORY.md` (2026-08-31, K-057) and
`docs/reviews/salesperson-tool-reliability-ml.md` §11/§14, both live n=20 regressions on
`mistralai/ministral-3-3b` through a `LoggingToolRegistry` harness against real LM Studio.

Iteration 1 (two sentences: an inclusive-bound translation rule + a "don't state a conclusion
before your last planned tool call returns" rule) shipped: the targeted rounding defect went to
20/20 correct and **net full-reply correctness was 16/20 (80%)**. Iteration 2 added two more
sentences aimed at a *newly discovered* third defect (silent category-omission at synthesis time).
Result: the targeted defect did not improve (30% wrong vs. the shipped version's own 20%), the
model still passed `category` 0/20 times — and **net correctness fell to 14/20 (70%)**, because the
added guidance suppressed a multi-call self-correction pattern the model had been performing on its
own under the shorter wording (a second, category-scoped `filter_products` call that §14.4 measured
rescuing ~two-thirds of the reps it fired on). Iteration 2 was reverted, never shipped.

**Consequence — two rules for any prompt/tool-description wording eval:**

1. **Score net task correctness on every iteration, not just the targeted defect rate.** A wording
   change is a global intervention on the model's behavior, not a patch to one code path: it can
   fix its target and still lose ground overall. An eval that measures only the defect being fixed
   cannot see that, and will ship a regression as a win.
2. **Guidance can remove an emergent behavior that was already helping.** Before adding a sentence,
   check whether the current sample shows the model self-correcting; if it does, that behavior is
   part of the baseline you must not lose, and its rate belongs in the scorecard alongside the
   defect rate. (`docs/reviews/…-ml.md` §11.5 had already warned of this in the abstract — steering
   the model toward single-tool use "could plausibly remove the self-correction path"; iteration 2
   is the measured confirmation.)

Model-specific: measured only on `mistralai/ministral-3-3b` at n=20 per arm — treat the *direction*
as a design caution, not the effect size as portable.

**Context:** `falkor-chat` K-057 (salesperson demo agent tool-call reliability wording fix).
