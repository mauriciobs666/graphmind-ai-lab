# Local model + embedding selection under a 16 GB RAM budget — method note

> **Type:** data-scientist method note (advisory — no code/config changed here).
> **Author:** data-scientist · **Date:** 2026-07-18
> **Decision it serves:** the user's machine was downgraded **32 GB → 16 GB** (it was crashing under
> overload). Pick a chat-model + embedding-model configuration that fits **16 GB shared** across
> Windows + WSL2/Docker + FalkorDB + LM Studio, with headroom, before resuming K-022 (the LLM-native
> workflow executor) — which leans harder on the chat model (tool-calling, multi-step instruction
> following) than plain chat did.
> **Consumes:** `docs/plans/m3-executor.md` (executor demands), `docs/plans/m3-executor-ml.md`
> (guard/tool-calling method), `server/falkorchat/config.py` + `server/.env.example` (current config).

---

## TL;DR recommendation

**Keep both current models. The downgrade does not force a model change — it forces discipline on
context length and KV-cache, plus leaving explicit headroom for FalkorDB and the OS.** The 32 GB→16 GB
crash was almost certainly over-allocation (32 GB-era context windows / no reserved headroom), not the
models being intrinsically too big. The current pair resident together is ~5 GB of an ~8 GB model
budget.

| Slot | Recommended | Quant | Context | Footprint (resident, est.) |
|---|---|---|---|---|
| Chat LLM | `qwen/qwen3-4b-2507` (unchanged) | **Q4_K_M** | **8192** (KV cache **Q8**) | ~3.5–4 GB |
| Embedder | `text-embedding-qwen3-embedding-0.6b` (unchanged) | Q8 | n/a | ~0.9–1 GB |
| **Both loaded** | | | | **~4.5–5 GB** |

**Do not drop the chat model below 4B** — that is the one change that would directly damage the K-022
executor (see §2). **Do not change the embedder** — the RAM saving is marginal (~0.4 GB) and it forces
a full `ws:acme` re-bootstrap + re-embed (see §3).

---

## 1. RAM budget breakdown (the load-bearing assumptions)

The 16 GB is **one physical pool**. LM Studio runs on the **Windows** side; FalkorDB/Docker/the app run
in **WSL2**. They do not have separate memory — they compete for the same 16 GB, so the model budget is
only a *fraction* of it. My assumed split (state your real numbers back if any are off):

| Consumer | Assumed reserve | Basis / assumption |
|---|---|---|
| Windows 11 host OS (idle + browser/LM Studio UI) | **~3.5 GB** | Typical Win11 idle ~3–4 GB; estimate. |
| WSL2 Linux VM + Docker engine baseline | **~2.0 GB** | WSL2 VM + dockerd + FalkorDB container process overhead (excl. graph data). Estimate. |
| **FalkorDB in-memory graph** (`ws:acme` + `reference` + `identity`) | **~1.0 GB** | Demo/proof scale is MBs today, but RAM is FalkorDB's binding constraint and the executor adds run/step-run + (debug) TraceEvent hot-growth (`m3-executor.md` §3). Reserve for growth. |
| **Safety headroom** (avoid the crash that triggered this) | **~1.5 GB** | The explicit buffer the 32 GB config lacked. Non-negotiable given the machine was already crashing. |
| **→ Left for LM Studio models** | **~8 GB** | 16 − (3.5 + 2.0 + 1.0 + 1.5). This is the real model budget. |

**Conclusion:** ~8 GB for models. The recommended pair uses ~5 GB of it → ~3 GB of slack. That slack is
the point — it is what keeps the machine off the crash boundary while FalkorDB grows and Windows does
Windows things.

> **Biggest uncertainty:** the WSL2 memory cap. By default WSL2 will balloon to ~50 % of host RAM (8 GB
> here) and is slow to release it back to Windows. If you have **not** set a `.wslconfig` `memory=` cap,
> WSL2 alone can transiently squeeze the Windows-side LM Studio budget and *reproduce the crash even with
> small models*. See §6 OQ-1 — this may matter more than any model choice.

---

## 2. Chat model — keep Qwen3-4B, cap context, quantize KV

**Verdict: the current `qwen/qwen3-4b-2507` (Qwen3-4B-Instruct-2507, non-thinking) fits comfortably and
should stay.**

- **Weights:** `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` = **2.5 GB** on disk/resident
  ([bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF),
  [unsloth](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)).
- **KV cache** (the lever that actually varies with your settings): Qwen3-4B is 36 layers, GQA with 8
  KV heads ([apxml](https://apxml.com/models/qwen3-4b)). KV cost ≈ `2 × layers × kv_heads × head_dim ×
  bytes/token`. Depending on the exact head_dim used by the runtime this lands **~0.15–0.36 MB/token**;
  at **8K context** that is **~1.2–2.9 GB** at fp16, roughly **halved with Q8 KV cache**. This — not the
  weights — is what blew a 32 GB config that ran 32K+ contexts. **Cap context at 8192 and enable Q8 KV
  cache in LM Studio.**
- **Total resident (weights + Q8 KV @8K + compute/runtime buffers):** **~3.5–4 GB**. Fits the ~8 GB
  budget with the embedder alongside.

**Is 8K context enough for the executor?** Yes. The executor assembles: node `systemPrompt` + run `ctx`
+ a thread window capped at `THREAD_CONTEXT_WINDOW = 20` messages + 1–2 tool schemas
(`m3-executor.md` §2.2, coordination doc D6). That is comfortably under 8K. If a future node needs a
larger fence, raise context deliberately and re-check the KV cost against the budget.

**Why not go smaller to save RAM?** Because the executor *is* a tool-calling agent loop, and
tool-calling is where small models fall off a cliff. Published BFCL-style figures put **4B ~50 %,
2B ~43 %** function-calling accuracy — a real, measured drop, not a rounding
([promptquorum](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)). The
K-022 note already flags 4B tool-calling as an *accepted risk* mitigated by native-primary +
JSON-structured-output fallback (`m3-executor-ml.md` Q3, D4). Dropping below 4B spends the one resource
the executor cannot spare to save ~1–1.5 GB the budget does not need. **4B is the floor here.**

**If you still want smaller alternatives** (ranked; only if §6 OQ-1 shows the budget is genuinely
tighter than assumed — otherwise keep 4B):

| Rank | Model | Quant → weights (est.) | Trade-off vs Qwen3-4B |
|---|---|---|---|
| **1 (fallback)** | `qwen/qwen3-4b-2507` @ **Q3_K_M** | ~2.0 GB | Same model, ~0.5 GB less, measurable quality/format-validity loss at Q3. Prefer shrinking **context/KV** before quant — Q4_K_M is the quality/size knee. |
| 2 | Qwen3-1.7B-Instruct | Q4_K_M ~1.1 GB | Saves ~1.4 GB weights but tool-calling reliability drops toward the 2B tier (~43 %). Only if the executor is demoted to structured-output-only prompting. Same family = same prompt behaviour, easiest swap (`FALKORCHAT_LLM_MODEL` only). |
| 3 | Gemma-3-4B-it | Q4_K_M ~2.6 GB | Comparable size to Qwen3-4B, no RAM win; different tool-call formatting → would need the §2.2 dual-shape parser re-verified. **No reason to switch given no RAM benefit.** |

Ranking rationale: staying in the **Qwen3 family** preserves the prompt/tool-call behaviour the
executor's dual-shape parser (`llm.py` `chat`) and the guard prompts were built and tested against.
Cross-family swaps re-open the D4 tool-calling risk for zero RAM benefit.

Swapping the chat model is a **free config change** (`FALKORCHAT_LLM_MODEL`) — no migration — so this
slot can be tuned freely if measurement disagrees with the estimates here.

---

## 3. Embedding model — keep it, do not touch the dimension

**Verdict: keep `text-embedding-qwen3-embedding-0.6b` at 1024-dim. Explicitly do not change it.**

- **Footprint:** the 0.6B GGUF is **~639 MB at Q8** ([Qwen HF card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF),
  [Simon Willison](https://simonwillison.net/2025/Jun/8/qwen3-embedding/)); ~0.9–1 GB resident with
  runtime overhead. It is already the small partner in the pair — there is almost nothing to save here.
- **Quality:** MTEB-multilingual **64.33**, competitive with models several times its size, 32K context,
  native 1024-dim ([Qwen HF card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
  [morphllm comparison](https://www.morphllm.com/ollama-embedding-models)). For a chat-message GraphRAG
  corpus this is strong; there is no retrieval-quality reason to move.
- **Migration cost if you changed it (why not to):** a different embedder almost always means a different
  output **dimension**, which forces:
  1. re-bootstrapping the `ws:acme` vector index at the new dim (`bootstrap_schema.sh` with
     `EMBEDDING_DIM=`), and
  2. **re-embedding every existing Message** (the `EmbeddingWorker` path), because
  3. FalkorDB **silently accepts a wrong-dim `vecf32`** and then drops it out of the ANN index — the
     retrieval *passes while finding nothing* (AGENTS.md; `.env.example:16`). This is the single nastiest
     failure mode in this system and it fails **silently**.

  The RAM saving that would buy is ~0.4 GB of model weights plus a marginal per-Message `vecf32` saving
  (a lower dim shrinks the inline vector on every Message — real but small at proof scale). **Not worth a
  silent-corruption-class migration.** The dimension is also a per-workspace FalkorDB RAM line (rule 6),
  so if you *ever* re-bootstrap `ws:acme` for another reason, that is the moment to reconsider dim — not
  now, and not for a 0.4 GB reason.

**One correctness note carried from the codebase:** `FALKORCHAT_EMBEDDING_DIM` **must** equal the model's
real output dim (1024) and match how `ws:acme` was bootstrapped. The config default is `1536`
(`config.py:36`) — the *served* app overrides it to 1024 (`.env.example:19`, `start_server.sh`). Keep
that override; a mismatch is the silent-ANN-drop trap above, not a crash.

---

## 4. Concurrency reality — both models resident at once

The chat model and embedder are used **concurrently within a single workflow run**: the research node
calls `graphrag_retrieve`, which **embeds the query** (`tools.py` → `embedder.embed`) *and then* the LLM
reasons over the seeds (`m3-executor.md` §4, M5). Embeddings are also computed out-of-band by the
`EmbeddingWorker` on every posted message. So both models are on the hot path and **both should be
resident simultaneously** — budget for the sum, not the max.

- **Both loaded:** ~4.5–5 GB (§2 + §3). Fits the ~8 GB budget → keep both loaded (LM Studio JIT with a
  long/့disabled idle-TTL). This is the recommended posture: it avoids per-call load latency, and the
  pair fits.
- **On-demand (evict-idle) loading** — LM Studio can auto-unload the idle model and reload on demand.
  This trades ~2.5 GB of resident RAM for a **multi-second reload stall** every time the run alternates
  between embedding and chat (which the research node does *within one step*). Given the pair already
  fits, on-demand loading would add latency and cold-start jitter for no benefit. **Only fall back to it
  if §6 OQ-1 reveals the real budget is <6 GB.** If forced there, prefer keeping the **embedder** resident
  (tiny, called first in the research step) and evicting the chat model between runs.

---

## 5. Concrete recommended configuration

```
# Chat (free to change — FALKORCHAT_LLM_MODEL)
FALKORCHAT_LLM_MODEL=qwen/qwen3-4b-2507        # unchanged
  quant:    Q4_K_M                              # ~2.5 GB weights
  context:  8192                                # LM Studio context length
  KV cache: Q8                                  # halves KV footprint; set in LM Studio

# Embedder (do NOT change — forces ws:acme re-bootstrap + re-embed)
FALKORCHAT_EMBEDDING_MODEL=text-embedding-qwen3-embedding-0.6b   # unchanged
FALKORCHAT_EMBEDDING_DIM=1024                                    # MUST match ws:acme bootstrap
  quant: Q8                                     # ~0.64 GB weights

# LM Studio: keep BOTH models loaded (disable/lengthen idle auto-unload)
# WSL2: cap the VM so it cannot starve the Windows-side model budget (see OQ-1)
```

Resident total ≈ **~5 GB models** + ~1 GB FalkorDB + ~5.5 GB OS/WSL2/Docker + ~1.5 GB headroom ≈ **~13
GB of 16 GB**, leaving a deliberate buffer.

**Fallback ladder if the machine still struggles** (apply in order, cheapest-first — each is reversible):
1. **Cap WSL2 memory** via `.wslconfig` (`memory=8GB`, `swap=…`) — likely the real fix (§6 OQ-1). Do
   this *first*; it may resolve everything without touching models.
2. **Drop chat context 8192 → 4096** and confirm Q8 KV cache is on — biggest model-side RAM lever, minimal
   quality cost given the ≤20-message thread window.
3. **Enable LM Studio evict-idle loading** (accept reload latency, §4).
4. **Chat weights Q4_K_M → Q3_K_M** (~0.5 GB, measurable quality cost — verify format-validity stays
   ≥ ~90 % per `m3-executor-ml.md` Q3 before trusting it in the executor).
5. **Last resort:** chat → Qwen3-1.7B (§2 rank 2) — only with the executor demoted to structured-output
   prompting, and re-run the guard/tool calibration.

Do **not** reach for the embedder in this ladder — it is the smallest resident model and the most
expensive to change.

---

## 6. Open questions & risks (confirm before acting — I run isolated and cannot ask live)

- **OQ-1 (highest leverage — likely the actual root cause):** Is there a `%UserProfile%\.wslconfig` with a
  `memory=` cap? If not, WSL2 balloons to ~8 GB and is slow to release it, which can starve the
  Windows-side LM Studio budget and **reproduce the crash regardless of model size**. Please check /
  set this *before* concluding any model is "too big." This is a devops action, not an ML one — flag to
  the `devops` agent.
- **OQ-2 (verify, don't trust my estimates):** The OS/WSL2/Docker reserves in §1 are assumptions. The
  cheap ground-truth: with FalkorDB + the app running but LM Studio unloaded, read actual free RAM
  (Windows Task Manager + `free -h` in WSL). If the non-model reserve is materially above ~6.5 GB, the
  model budget shrinks and the §5 fallback ladder starts at step 1–2. `scripts/load_test.sh` already
  captures a per-workspace FalkorDB RAM delta — use it to pin the FalkorDB line rather than my 1 GB
  guess.
- **OQ-3 (KV-cache figure):** the per-token KV estimate (0.15–0.36 MB) spans the runtime's head_dim
  handling; I did not measure it on your LM Studio build. The *decision* (cap context, Q8 KV) holds
  either way, but the exact resident number should be read from LM Studio's model-load readout, not
  trusted from this note.
- **OQ-4 (FalkorDB growth):** the 1 GB FalkorDB reserve is proof-scale. If `ws:acme` accumulates real
  chat history + debug `TraceEvent` runs, that line grows and eats headroom. Re-check the split when the
  graph gets loaded up — RAM is FalkorDB's binding constraint (AGENTS.md rule 6).
- **Risk — silent embedding mis-config, not a crash:** if anyone "optimises" by changing the embedder or
  `FALKORCHAT_EMBEDDING_DIM` without re-bootstrapping `ws:acme` + re-embedding, retrieval goes silently
  empty (§3). Any embedder change must route through `graph-dba` (index re-bootstrap) + a re-embed pass —
  it is not a config-only change like the chat model is.
- **Risk — model IDs/footprints are perishable:** sizes cited are current GGUF builds (2026-07); LM Studio
  catalog ids and quant sizes drift. Verify the exact file size in LM Studio at download time; the
  *budget logic* (§1) is what to keep, not the specific byte counts.

---

## Evaluation — how to prove the chosen config is right

A RAM recommendation is only half a deliverable without an acceptance check. Two cheap gates:

1. **Fit gate (measured, not estimated).** With the §5 config loaded and a workflow run driving the
   research node (embed + chat in one step), read peak RAM across Windows + WSL. **Acceptance:** total
   stays **≤ ~14 GB** (i.e. ≥ ~2 GB free) through a full triage run, and no OOM/crash across 3
   consecutive runs. This is the direct regression guard for "did the downgrade break us."
2. **Quality-didn't-regress gate.** The config change must not degrade the executor. Reuse the K-022
   instruments: the guard **golden set** `server/tests/eval/golden_guards.jsonl` (calibration gate:
   false-advance ≤ 10 % AND advance-recall ≥ 0.80, per `m3-guard-calibration.md`) and the live triage
   e2e `test_workflow_live.py` (AC-1…AC-4). **Acceptance:** the guard gate still passes and the live flow
   still reaches `done` at Q4_K_M / 8K context. If Q4_K_M/8K silently drops tool-call format-validity
   below the `m3-executor-ml.md` Q3 ~90 % floor, that shows up as guard/tool failures here *before* it
   ships — which is the whole reason those instruments exist.

Both gates are already-built harnesses; this note asks only that they be **re-run once at the new
config** as the acceptance for the RAM change, not that anything new be constructed.

---

## 7. Addendum — Mistral / Ministral family as chat-LLM alternatives (2026-07-18)

> **Why this section exists:** §2 kept Qwen3-4B but only compared it against *smaller Qwen/Gemma*
> fallbacks — it never evaluated the Mistral/Ministral family the user asked about. This addendum
> does the head-to-head, on current benchmarks and current GGUF footprints, against the **same
> constraint** §1 established: ~8 GB total model budget, embedder taking ~1 GB, leaving **~4–5 GB
> resident for the chat model** (weights + KV cache + context). Nothing in §1–§6 changes; this
> either confirms the incumbent or names a challenger under the same ceiling. The decisive axis is
> unchanged: **function/tool-calling reliability for the K-022 executor**, not chat fluency.

### 7.1 The candidates, measured against the ~4–5 GB chat slot

| Model | Params | Weights Q4_K_M | Resident @8K + Q8 KV (est.) | Fits ~4–5 GB w/ embedder? | Native tool-calling | License |
|---|---|---|---|---|---|---|
| **Qwen3-4B-2507** (incumbent) | 4B | **2.5 GB** | ~3.5–4 GB | **Yes** (~3 GB slack, §1) | Yes | Apache 2.0 |
| Ministral 8B **2410** | 8.02B | **4.91 GB** | ~6.5–7.5 GB | **No** — weights ≈ ceiling *before* KV | Yes | **Mistral Research (non-commercial)** |
| Ministral 3B **2410** | 3B | — | — | **N/A — no open weights** (API-only in 2024) | Yes (API) | Mistral Research |
| Mistral **7B v0.3** | 7.25B | **4.37 GB** | ~5.5–6.5 GB | **No / marginal** — weights alone crowd the embedder out | Yes (`TOOL_CALLS` tokens) | Apache 2.0 |
| Mistral **Small 3.2** 24B | 24B | **~15 GB** | ~17+ GB | **No — 3× over the whole model budget** | Yes (strong) | Apache 2.0 |
| **Ministral 3 8B (2512)** | 8B | **5.2 GB** | ~6.5–7.5 GB | **No** — same 8B-tier RAM wall | Yes ("best-in-class agentic", JSON) | **Apache 2.0** |
| **Ministral 3 3B (2512)** | 3B | **~2.0 GB** *(est.)* | ~3–3.5 GB | **Yes** | Yes ("best-in-class agentic", JSON) | **Apache 2.0** |

Footprints: Ministral 8B-2410 Q4_K_M = 4.91 GB (bartowski), Mistral 7B v0.3 Q4_K_M = 4.37 GB
(MaziyarPanahi/Ollama), Mistral Small 3.2 24B Q4_K_M ≈ 15 GB (unsloth/bartowski), Ministral 3 8B-2512
Q4_K_M = 5.2 GB (unsloth). The Ministral 3 **3B**-2512 Q4_K_M size is **estimated** at ~2.0 GB by
analogy to the 3B/Q4_K_M class (Qwen3-4B is 2.5 GB) — **verify the exact file in LM Studio at
download time** (`bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF`).

**RAM reads the verdict before quality does.** Everything at the **8B tier** — Ministral 8B-2410,
Ministral 3 8B-2512 — carries **~5 GB of weights alone**, which is the *entire* chat slot *before*
one token of KV cache or the resident embedder. Under §1's budget these do **not** co-reside with the
embedder; you'd have to (a) drop to Q3 (quality loss, and the §2/OQ format-validity risk), or (b) run
LM Studio evict-idle and eat a multi-second reload every time the research node alternates embed↔chat
within a single step (§4). Both spend real quality/latency to run a model that isn't clearly better on
the axis that matters. **Mistral Small** (24B, ~15 GB) is out on size, full stop. **Mistral 7B v0.3**
technically fits at ~4.37 GB but leaves almost nothing for the embedder+KV and is a **2024-era 7B**
whose tool-calling is not competitive with a *modern* 4B (below). **Ministral 3B-2410 has no open
weights** — it was API/commercial-only in 2024, so it cannot be served in LM Studio at all. The only
Mistral candidate that both **fits the budget** and is **current + cleanly licensed** is the brand-new
**Ministral 3 3B (2512)**.

### 7.2 Quality on the decisive axis — tool-calling

The honest state of the evidence:

- **Qwen3-4B-2507 has a concrete, published tool-calling number:** BFCL-V4 **overall 62.04 %**
  (live 75.52 %, non-live 82.58 %, multi-turn 35.25 %). That is the incumbent's proven floor and the
  one directly-comparable data point in this whole comparison.
- **No Ministral/Mistral model in this list has a comparable public BFCL score I could verify.**
  Ministral 8B-2410's model card cites an *internal* function-calling benchmark of **31.6** — but that
  is Mistral's own scale, **not** BFCL, so it is **not** comparable to Qwen's 62.04 and I will not
  pretend it is. General benchmarks that *are* comparable: MMLU — Qwen3-4B-2507 (low-70s per its card,
  not re-verified here) vs Ministral 8B-2410 **65.0** vs Ministral 3 8B-2512 **76.1**. General
  capability rises with the 8B models; that is expected and not the question.
- **The sub-4B cliff §2 warned about still applies to the one Mistral that fits.** Ministral 3 3B is a
  3B model. §2's whole argument — function-calling falls off sharply below 4B (BFCL ~50 % at 4B →
  ~43 % at 2B) — is a *generic* small-model finding. Mistral markets the 2512 family as
  "best-in-class agentic capabilities with native function calling", which is a *claim it might beat
  the generic cliff*, but there is **no BFCL number to confirm it yet** (the family is ~6 weeks old).
  That is exactly the gap an A/B test closes (§7.5).

### 7.3 Head-to-head verdict vs. Qwen3-4B

**The incumbent Qwen3-4B-2507 still wins, on evidence, not defaults.** No Mistral/Ministral option
*both* fits the ~4–5 GB chat slot *and* has demonstrated tool-calling parity:

- The **quality-competitive** Mistrals (Ministral 8B-2410, Ministral 3 8B-2512, Mistral Small) **bust
  the RAM budget** — the 8B tier by ~1 GB of weights before KV, Mistral Small by 3×. RAM decides them
  out before quality is even argued.
- The Mistral that **fits the budget cleanly** (Ministral 3 3B-2512) sits **below the 4B tool-calling
  floor** §2 established, with **no published BFCL** to show it beats the generic sub-4B cliff. It is
  *promising*, not *proven*.
- **Mistral 7B v0.3** is the trap option: it looks like "a bigger model for similar RAM", but it's a
  2024 7B whose tool-calling is weaker than a 2025 4B, and it still crowds the embedder. Larger ≠
  better here.

So: **keep Qwen3-4B-2507 as the recommended chat model** (§2 stands unchanged). The Mistral family does
not offer a strictly-better swap under 16 GB — it offers **one genuinely interesting experiment**
(§7.5), not a decision the benchmarks already make for you.

### 7.4 Licensing note

Relevant even in a dev-phase project, because a research-only license is a **dead end** the moment this
becomes anything commercial — better to know now:

- **Ministral 8B / 3B 2410** → **Mistral Research License (non-commercial only).** "Research Purposes"
  explicitly *excludes* use by employees/contractors in daily tasks and any activity intended to
  generate revenue, including proof-of-concept. A dev-phase POC that's a step toward a product is
  arguably already outside it. **Avoid building on the 2410 Ministrals** unless you're prepared to
  re-license or swap later.
- **Ministral 3 (2512) family — 3B/8B/14B — is Apache 2.0.** This is the important change: it removes
  the licensing objection entirely and is why the 2512 3B, not the 2410 8B, is the Mistral candidate
  worth testing.
- **Mistral 7B v0.3** is Apache 2.0. **Mistral Small 3.2** is Apache 2.0 (but out on size).
- **Qwen3-4B-2507** (incumbent) is Apache 2.0 — no change, no concern.

### 7.5 The honest "try it" recommendation

If the user wants to *actually test* a Mistral rather than take "incumbent wins" on faith — which is
reasonable given the 2512 family is too new to have BFCL coverage — there is **one clean try candidate**:

**Try #1 — Ministral 3 3B (2512), Apache 2.0.** It's the only Mistral that fits the §1 budget with
slack, has a clean license, and ships native FC + JSON output with an explicit agentic pitch. The
whole question is empirical: *does Mistral's small-model agentic tuning beat the generic sub-4B
tool-calling cliff?* Don't decide that on the vendor card — **run it through the K-022 guard
golden-set gate** (§Evaluation gate 2). If it clears **false-advance ≤ 10 % AND advance-recall ≥ 0.80**
(`m3-guard-calibration.md`) *and* the live triage e2e (`test_workflow_live.py`, AC-1…AC-4) still
reaches `done`, it's a legitimate co-recommendation that also frees ~0.5–1 GB of headroom. If it
misses the guard gate, that's the sub-4B cliff confirming itself on *your* workload — fall back to
Qwen3-4B and you've lost only a download.

**Try #2 (conditional) — Ministral 3 8B (2512), Apache 2.0** — *only* if the user is willing to run
LM Studio evict-idle loading or drop chat context to make ~5 GB of weights fit (§4). This is the
"is a well-licensed 8B worth the RAM/latency cost?" test. I rank it **below** Try #1 because it fights
the RAM budget this whole note exists to respect; run it only if Try #1's 3B misses the guard gate and
you want an 8B-class Mistral before returning to Qwen.

**Do not** A/B Mistral 7B v0.3 (older, weaker tool-calling for the RAM), the 2410 Ministrals
(non-commercial + 8B RAM wall / 3B unavailable), or Mistral Small (3× over budget).

### 7.6 Concrete config if Ministral 3 3B is adopted

Swapping the chat model is a **free config change** — `FALKORCHAT_LLM_MODEL` only, no re-bootstrap, no
re-embed, **zero migration cost** (unlike the embedder, §3). So this can be trialed and reverted at
will. Slotting into §5's recommended-config block, the chat line becomes:

```
# Chat — Ministral 3 3B trial (free swap; revert by restoring the Qwen line)
FALKORCHAT_LLM_MODEL=mistralai/ministral-3-3b-instruct-2512   # verify exact LM Studio model id at download
  GGUF repo: bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF (or mistralai/…-GGUF)
  quant:    Q4_K_M                                            # ~2.0 GB weights (VERIFY actual size in LM Studio)
  context:  8192                                              # cap here despite the model's 256k — KV budget, §2
  KV cache: Q8                                                # same discipline as the incumbent
```

Everything else in §5 is unchanged — the embedder is untouched (§3's silent-ANN-drop trap is why),
and the two §Evaluation gates are the acceptance for the swap exactly as written. The dual-shape
tool-call parser (`llm.py` `chat`) must be **re-verified** against Mistral's tool-call format before
trusting the swap in the executor — Mistral emits `TOOL_CALLS`-token / JSON shapes that may differ from
Qwen's, which is precisely what the guard golden-set gate will surface. Route any parser adjustment to
`coder` with this note + `m3-executor-ml.md` §2.2 as the brief.

### 7.7 Open questions folded in (I run isolated)

- **OQ-5:** The Ministral 3 3B-2512 Q4_K_M file size is an **estimate** (~2.0 GB). Confirm in LM Studio;
  the *decision* (it's the only fitting Mistral) holds across the plausible 1.9–2.3 GB range, but the
  slack math in §1 should use the real number.
- **OQ-6:** No public BFCL/IFEval for **any** 2512 Ministral yet (family is ~6 weeks old at this
  writing). The Try #1 recommendation is deliberately structured so the K-022 guard gate *is* the
  measurement — do not swap on the vendor "best-in-class agentic" claim alone.
- **OQ-7:** The Qwen3-4B-2507 MMLU figure (low-70s) was not re-verified in this pass; it's cited only
  as directional context. The **BFCL 62.04 %** for Qwen3-4B *was* fetched and is the number the
  decision rests on.

---

### Sources
- Qwen3-4B-Instruct-2507 Q4_K_M size — [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF), [unsloth GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
- Qwen3-4B architecture (layers/GQA/KV heads) — [apxml](https://apxml.com/models/qwen3-4b), [HF model card](https://huggingface.co/Qwen/Qwen3-4B)
- Small-model tool-calling reliability (4B vs 2B BFCL) — [promptquorum](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)
- Qwen3-Embedding-0.6B size / dims / MTEB — [Qwen HF card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), [Simon Willison](https://simonwillison.net/2025/Jun/8/qwen3-embedding/), [morphllm embedding comparison](https://www.morphllm.com/ollama-embedding-models)
- Qwen3-4B BFCL-V4 tool-calling scores (62.04 % overall) — [llm-stats BFCL-V4](https://llm-stats.com/benchmarks/bfcl-v4), [Gorilla BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- Ministral 8B-2410 specs / MMLU 65.0 / internal FC 31.6 / Mistral Research License — [Mistral HF card](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410); Q4_K_M 4.91 GB — [bartowski GGUF](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF)
- Mistral Research License (non-commercial) terms — [Mistral Help Center](https://help.mistral.ai/en/articles/347393-under-which-license-are-mistral-s-open-models-available), [MNPL](https://mistral.ai/news/mistral-ai-non-production-license-mnpl/)
- Ministral 3B-2410 not open-weighted (API-only in 2024) — [HF discussion](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/discussions/4), [Mistral docs model card](https://docs.mistral.ai/models/model-cards/ministral-3b-24-1)
- Mistral-7B-Instruct-v0.3 Q4_K_M 4.37 GB / native tool tokens — [MaziyarPanahi GGUF](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf), [Ollama v0.3](https://ollama.com/library/mistral:7b-instruct-v0.3-q4_K_M)
- Mistral Small 3.2 24B Q4_K_M ~15 GB / 13.4 GB VRAM — [unsloth GGUF](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF), [willitrunai](https://willitrunai.com/blog/mistral-models-gpu-requirements)
- Mistral 3 / Ministral 3 (2512) family — Apache 2.0, Dec 2 2025, 3B/8B/14B — [Mistral news](https://mistral.ai/news/mistral-3/); 8B MMLU 76.1 / Q4_K_M 5.2 GB / native FC + JSON / 256k ctx — [unsloth Ministral-3-8B GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF); 3B GGUF — [bartowski](https://huggingface.co/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF), [mistralai](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF)
</content>
</invoke>
