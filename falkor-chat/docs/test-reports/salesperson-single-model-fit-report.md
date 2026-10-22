# Salesperson single-model — live GPU-fit check

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-066

## Summary

Live check, 2026-10-21, on the RTX 4050 Laptop (6141 MiB, WDDM). The question was whether
`mistralai/ministral-3-3b` (`-c 8192 --gpu max`) plus `text-embedding-qwen3-embedding-0.6b` fit
fully on the GPU, with no CPU layers and no spill. **Verdict: fits with spill (WDDM
oversubscription), not fully.** All layers are on the GPU and ministral generates at full speed
(38.5–40.5 tok/s against the bench's 38.1–38.9). But the two models' combined working set is
larger than VRAM. Windows pages about 1.6 GiB of ministral out whenever the embedder runs, and
pages it back for the next chat. `nvidia-smi` stays under total the whole time (5839/6141 MiB).
The swap costs roughly 0.1 s of ministral TTFT and 0.25–1.4 s on the first embedding after a
chat. A `--gpu 0.5` contrast (14.0–14.1 tok/s, TTFT 2.1–2.3 s) shows that the throughput
instrument tells real CPU offload apart clearly. Separately, the unified 8192-token KV pool
rejects **all three** of three concurrent ~3.1k-token requests with "Context size has been
exceeded"; two concurrent requests succeed.

CPG: not applicable — a live hardware/runtime capacity measurement, with no code under test.

## Environment and baseline

- Run date 2026-10-21. GPU: NVIDIA GeForce RTX 4050 Laptop GPU, WDDM, `memory.total` 6141 MiB.
  LM Studio runtime `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.59.0` (the model-bench reference
  runs used runtime 2.40.0).
- State on arrival: `qwen/qwen3-4b-2507` loaded (8192 ctx, PARALLEL 4), `memory.used` 3766 MiB.
- After `lms unload --all`: no models loaded, `memory.used` **60 MiB / 6141 MiB**.
- Estimates (`lms load ... --gpu max --estimate-only`): ministral `-c 8192` **4.84 GiB** GPU;
  `text-embedding-qwen3-embedding-0.6b` **853.36 MiB** (Confidence: LOW).
- **Spill instrument.** `nvidia-smi` can't see WDDM shared-memory spill (review M3), so this run
  also reads the Windows performance counters `\GPU Process Memory(*)\Dedicated Usage` and
  `\GPU Process Memory(*)\Shared Usage` for the NVIDIA adapter (LUID `0x00011343`), per process,
  via `powershell.exe Get-Counter`. Shared Usage on `llama-server` is the direct spill signal.
- Reference throughput: model-bench run records for `mistralai/ministral-3-3b` (Q8_0, ctx 8192,
  runtime 2.40.0, no other resident model) — median `tokensPerSecond` **38.9**
  (`chat-responder-grounded-answers-...-2026-09-20T01:41:47Z.json`, 30 calls), **38.9**
  (`guard-judge-understanding-...-2026-09-20T01:34:52Z.json`, 85 calls), **38.1**
  (`nlq-structured-query-...-2026-09-20T01:36:40Z.json`, 40 calls). The ~38 tok/s figure holds.

## Full-GPU load (ministral `-c 8192 --gpu max` + embedder)

**ministral alone** — `lms load mistralai/ministral-3-3b -c 8192 --gpu max -y`: loaded in 6.28 s,
CLI reports 4.18 GiB; `lms ps` SIZE 4.49 GB, CONTEXT 8192, PARALLEL 4, DEVICE `Local` (no offload
column). Runtime log (`lms log stream -s runtime`):

```
W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999999, abort
I srv    load_model: loaded multimodal model, '...mmproj-Ministral-3-3B-Instruct-2512-F16.gguf'
I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 8192, kv_unified = 'true'
```

So `--gpu max` forces all layers (n_gpu_layers 999999) and llama.cpp's own fitter, which would
otherwise have reduced offload, declined to act. The F16 vision projector (mmproj) is loaded too —
the `ministral-3-3b` key is a VLM. `nvidia-smi` 5412 MiB; counters: `llama-server` (pid 17268)
dedicated **5353 MiB**, shared **112 MiB**.

**+ embedder** — `lms load text-embedding-qwen3-embedding-0.6b --gpu max -y`: loaded in 3.34 s
(609.54 MiB), runs inside the LM Studio process (pid 11136), log: `Batch size (2048) is < context
length (8192). Resetting batch size to context length`. Immediately after:

| process | dedicated | shared |
|---|---|---|
| adapter total | 5844 MiB | 661 MiB |
| `llama-server` 17268 (ministral) | **3689 MiB** (was 5353) | 176 MiB |
| LM Studio 11136 (embedder) | 2130 MiB | 484 MiB |

`nvidia-smi` `memory.used` **5839 / 6141 MiB** — under total, exactly as review M3 predicted, while
~1.66 GiB of ministral's allocations had been evicted from VRAM by WDDM to make room for the
embedder. The two models **do not fit side by side**; they fit only by WDDM paging one out while
the other runs (see Throughput for the swap observed per request).

## Throughput

Harness: scratchpad `bench.py` → `POST /api/v0/chat/completions`, temperature 0, `max_tokens` 256,
system prompt = a synthetic bakery catalog with a per-request nonce at the very start (defeats
prefix caching), user asks for three recommendations. Prompt ~3,100–3,170 tokens (a bit over the
2k target; larger than a typical turn, so conservative).

**Round 1 — 4 sequential requests** (embedder loaded, idle):

| # | prompt tok | completion tok | tok/s | TTFT s |
|---|---|---|---|---|
| 1 | 3142 | 221 | 38.51 | 2.106 (first after load) |
| 2 | 3106 | 256 | 38.65 | 0.759 |
| 3 | 3157 | 256 | 39.08 | 0.769 |
| 4 | 3167 | 256 | 38.73 | 0.765 |

After round 1: `llama-server` dedicated back to **5293 MiB**, embedder process down to **547 MiB**
dedicated (its 484 MiB shared gone) — WDDM paged ministral back in and the embedder out.

**Embedding** (`POST /api/v0/embeddings`, one short Portuguese sentence, 1024 dims, HTTP 200):
first call after chat **1.385 s**; in a second batch of 5: **0.260 s**, then 0.043, 0.028, 0.027,
0.026 s. After the embeddings: `llama-server` dedicated **3693 MiB** again, embedder 2136 MiB —
the swap reversed.

**Round 2 — 2 sequential requests right after the embeddings** (ministral partly paged out at
start): tok/s **38.64**, **40.53**; TTFT **0.836**, 0.753 s. Afterwards `llama-server` 5293 MiB
again.

Reading: generation speed is at full GPU speed (38.5–40.5 tok/s vs. the bench's 38.1–38.9) because
WDDM pages ministral's evicted ~1.6 GiB back in before decoding runs; the cost lands as a swap on
each model switch — roughly +0.07–0.1 s on ministral's TTFT and 0.25–1.4 s on the first
embedding after a chat. A storefront turn alternates embedder (retrieval) and ministral, so it
would pay this swap every turn.

## Partial-offload contrast (`--gpu 0.5`)

`lms unload mistralai/ministral-3-3b`, then `lms load mistralai/ministral-3-3b -c 8192 --gpu 0.5
--identifier ministral-half -y` (embedder still loaded). The load took 5.51 s. `llama-server`
dedicated **3423 MiB**, shared 112 MiB, and `nvidia-smi` 3992 MiB. The CPU half sits in ordinary
process RAM, which neither counter shows.

| # | prompt tok | completion tok | tok/s | TTFT s |
|---|---|---|---|---|
| 1 | 3142 | 249 | 14.10 | 2.290 |
| 2 | 3106 | 256 | 13.97 | 2.125 |
| 3 | 3157 | 256 | 14.10 | 2.158 |

Known CPU offload costs **~64%** of generation speed (14.0 vs 38.5–40.5 tok/s) and almost triples
TTFT, so the throughput check clearly tells the two cases apart. **Instrument caveat:** the
`llama-server` Shared Usage counter read 112 MiB in both the full and the half load. It reflects a
constant pinned host buffer, not CPU layers. Per-process *Dedicated* Usage is what moves, along
with tok/s. The half instance was then unloaded.

## Concurrency (optional)

Fresh `--gpu max -c 8192` ministral instance (`lms ps`: PARALLEL 4; log: `n_slots = 4,
n_ctx_slot = 8192, kv_unified = 'true'`), same ~3.1k-token prompts sent concurrently:

- **2 concurrent:** both succeeded. 36.27 / 33.43 tok/s, TTFT 1.338 / 1.057 s.
- **3 concurrent, run twice:** **all 3 requests failed both times**, with HTTP 400 `Engine protocol
  predict stream returned an error: {"code":500,"message":"Context size has been exceeded."}`.
  The 8192 tokens are one pool shared across slots, and 3 × ~3.1k exceeds it. Every request
  failed, not just the last one to arrive.
- One request immediately after: OK, 40.42 tok/s, TTFT 0.781 s, so the server recovers.

This matters if the demo can ever have three storefront turns in flight at once. Two is the safe
ceiling at this prompt size.

## Verdict

**Fits with spill: the pair is oversubscribed rather than fully resident.** No ministral layer is
on the CPU, and steady-state generation matches the bench (38.5–40.5 tok/s vs 38.1–38.9). But
ministral's process alone takes 5353 MiB dedicated of 6141. Adding the embedder (~2.1 GiB
dedicated while active, because its batch is reset to its 8192 context) forces WDDM to evict
~1.6 GiB of ministral to system RAM. It swaps back on the next chat. llama.cpp's own fitter warned
at load (`failed to fit params to free device memory`) and was overridden by `--gpu max`. After
the final reload with the embedder already resident, ministral's process had 5101 MiB dedicated
plus **368 MiB shared**, versus 5353 + 112 on the first, clean load: the same total, with ~256 MiB
more now in system memory. Generation was still 40.42 tok/s.

What it means for the plan: the per-turn cost of running ministral alone is small in this
measurement, at about +0.1 s TTFT and ≤1.4 s on the first embedding after a chat. Throughput is
unaffected. Weigh that against the 2.5 GB `qwen/qwen3-4b-2507` this change would remove. That
leaves "worth implementing" a stakeholder call, but not on a "fully fits" premise. Two cheap
levers this run did not test could plausibly remove the oversubscription: load the embedder with
a smaller context (its batch follows context, which drove the 2.1 GiB), or use a ministral variant
without the F16 mmproj vision projector. Each needs its own measurement.

## End state

`lms ps`: `mistralai/ministral-3-3b` (8192, PARALLEL 4, loaded `--gpu max`) and
`text-embedding-qwen3-embedding-0.6b`, both IDLE. `qwen/qwen3-4b-2507` and `ministral-half` are
unloaded. `nvidia-smi` 5719/6141 MiB. `:8200` answered HTTP 200 and was not touched, and neither
FalkorDB nor any source/config file was touched. Temporary counter script removed from the Windows
temp directory. Note: `qwen/qwen3-4b-2507` was resident on arrival and is now unloaded. If
`:8200`'s demo config still routes turns to qwen, LM Studio JIT-loads it on demand, which
reintroduces the pressure measured here.
