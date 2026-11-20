# K-070 embeds-first order and K-102 embed timeout retry: live LM Studio measurement

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-070, K-102 (M5)

## Summary

- **Tested:** `background._schedule_chunk_processing` embeds-first order (K-070, uncommitted working tree) against HEAD's interleaved order, and `EMBED_CHUNK_RETRY.retry_timeouts=True` (K-102) against HEAD's retry policy. Date 2026-11-20, base commit `13d1ba30` plus the uncommitted working tree.
- **Method:** a throwaway uvicorn on :8301 per run, one throwaway workspace per run (`ws:k070<mode><arm><rep>`, deleted after each run), `FALKORCHAT_ENABLE_AGENT=1`, workflow off. 8 synthetic documents (fixed seed, 4 chunks each, 32 embeds and 32 extracts) ingested through the real REST path. Embedder `text-embedding-qwen3-embedding-0.6b`, extractor `qwen/qwen3-4b-2507`, LM Studio at `http://localhost:1234` (resolved by `opencode.local.json`). Old code was run from scratch copies of `falkorchat/` with `git show HEAD:` versions of `background.py` / `retry.py` swapped in, selected via `PYTHONPATH`. The repo tree was not touched.
- **Arms:** `<order>_<retry>`: `new_new` = working tree; `old_new` = old order, new retry (isolates K-070); `old_old` = HEAD (isolates the pair). Two ingest modes: `batch` = `POST /documents/batch` (one sequential background chain, the case HISTORY.md describes); `conc` = 8 concurrent `POST /documents` (8 parallel chains).
- **Verdict:**
  - **K-070 reduces model swaps: confirmed, large effect, in the one clean pair.** Batch mode, 8 docs: LLM loads 8 (embeds-first) vs 30 (interleaved); request alternations 15 vs 63; wall time 473 s vs 552 s (-14%). This matches the plan's "~2N to ~2 swaps per document" claim: per document, 2 swaps instead of ~2N (N=4 chunks here). The order is per document, not per batch, so a batch of 8 documents still swaps about once per document (8 qwen loads, 15 switches).
  - **Wall-time effect is real but modest and noisy:** about -14% in batch (two reps, both direction-consistent), about -20% to -4% in conc (478/566 s vs 601/589 s; the two new reps differ by 88 s, as large as the effect). Too small a sample to conclude for conc.
  - **K-102 (timeout retry): NOT demonstrated either way.** Zero embed timeouts, zero embed failures, zero retries occurred in any of 11 completed runs, including the HEAD baseline (`old_old`). The timeout condition was not reproduced here, so this run cannot show that the change prevents it.
  - **All completed runs ended 8/8 documents `ready`, 0 `failed`, 0 stuck.**
- **Sample size:** 2 reps per cell, 8 documents each, one machine, one shared LM Studio with other users. Too small for statistical claims; the swap-count effect is large and deterministic enough to stand, the wall times and the K-102 null are not.
- CPG: considered, not relevant: a live black-box measurement of a scheduling-order change; no call-graph question was involved.

## Results

Request alternation = number of embed/chat switches in the LM Studio server log's request sequence (a switch demands a model swap only if the other model is not resident). LLM loads = qwen load events in LM Studio `main.log` (only valid for the first pair; `main.log` stopped updating at 11:51:56, see surprises). Chats above 32 are foreign traffic (see Contamination) or extractor repair re-prompts.

| Run | Wall s | ready / failed / stuck | Embeds | Chats | Alternations | LLM loads (main.log) | Embed timeouts / embed fails / extract fails / retries |
|---|---|---|---|---|---|---|---|
| batch new_new #1 | 472.6 | 8/0/0 | 32 | 32 | 15 | 8 | 0/0/0/0 |
| batch old_new #1 | 552.0 | 8/0/0 | 32 | 35 | 63 | 30 | 0/0/0/0 |
| batch new_new #2 | 468.8 | 8/0/0 | 32 | 67 (foreign) | 15 | n/a | 0/0/0/0 |
| batch old_new #2 | 553.5 | 8/0/0 | 32 | 67 (foreign) | 63 | n/a | 0/0/0/0 |
| conc new_new #1 | 478.1 | 8/0/0 | 32 | 40 | 3 | 1 | 0/0/0/0 |
| conc new_new #2 | 566.1 | 8/0/0 | 32 | 40 | 3 | n/a | 0/0/0/0 |
| conc old_old #1 | 601.5 | 8/0/0 | 32 | 38 | 49 | 1 | 0/0/0/0 |
| conc old_old #2 | 588.6 | 8/0/0 | 32 | 38 | 49 | n/a | 0/0/0/0 |

Not counted: two earlier batch runs (`new_new` #2 / `old_new` #2, first attempt) were discarded to `contam_*` and re-run because a foreign `modelbench` session was active; one first run was discarded because LM Studio rotated its server log mid-run and the harness lost the window. The raw per-run data (result.json, LM Studio window logs, server logs) was session-local and has been discarded; the tables above are the record.

### Observations

- **Batch, embeds-first:** the request log shows 4 embeds, then 4 extracts, per document (`EEEE XXXX` x 8): 15 switches for 8 documents. **Batch, interleaved:** strict `E X E X ...` for all 32 chunks: 63 switches, 30 qwen loads. Median per-switch penalty in the clean pair is about 2 to 3 s of model load, consistent with the 80 s wall difference.
- **Conc mode:** with 8 parallel chains the old order is already mostly decoupled across documents, yet interleaving within each chain still produced 49 switches vs 3 for the new order. Both ended 8/8 `ready`.
- **No timeouts anywhere**, including in HEAD's code with concurrent embed and extract jobs. The embedder reload takes about 2 s against a 30 s embed timeout.

## Defects

None found in the code under test. Findings about the measurement and the claim are under Feedback.

## Coverage & gaps

- Covered: REST sequential chain (batch) and 8 parallel REST chains; both orders; both retry policies in conc (`old_old` vs `new_new`).
- **Not covered:**
  - **MCP path** (one daemon thread per job, no ordering guarantee). It is where `ws:agent-team` failed (hundreds of concurrent jobs), and where K-102's timeout condition most likely arises. K-070 explicitly does not help there; K-102's effect there is untested.
  - The `new_old` and `old_new` cells in conc mode (dropped for time and foreign load); the retry flip is therefore not isolated in conc.
  - A forced-timeout test of K-102 (for example stalling the embedder behind a long generation); only the unit tests prove the retry logic.
  - Larger N, longer chunk counts, and a quiet LM Studio.
- Residual risk: K-102 is unmeasured live; the K-070 swap effect is not demonstrated on the MCP path.

## Contamination and surprises

1. **LM Studio was not a one-model-at-a-time host for most of this run.** In the first batch pair the embedder and qwen swapped (poller saw `[embedder]` then `[qwen]`, qwen loaded 8 vs 30 times). From about 11:40 on, another session's model-bench run (`modelbench run --model qwen/qwen3-4b-2507 ...`, `laya@q8_0` loaded) left three models resident together (`laya@q8_0`, qwen, embedder), and the poller then saw 0 to 3 state changes per run. In those runs the request alternations are demands for a swap that did not occur. The swap-count conclusion therefore rests on the first pair only (runs `batch_*_1`) plus the request-sequence counts; the second pair's wall times (469 vs 554 s) still show the same -15% gap, which I cannot explain by swaps alone (possible cause: per-request model-switch cost on a resident-but-GPU-evicted model; not verified).
2. **Foreign traffic.** The same machine's `modelbench` guard-judge runs sent 2-message qwen chats during most runs (chat counts of 38 to 69 vs 32 expected; `contamination.txt` shows modelbench was active at the end of every run after the first). Foreign chats did not change the embed counts or document outcomes, and the alternation counts matched the clean runs exactly in batch mode (15 and 63), but they inflate "Chats" and may add noise to wall time.
3. **`main.log` stopped updating** at 11:51:56 (last write), so LLM-load counts exist only for runs before it. Server logs in `~/.lmstudio/server-logs` do not contain load events at all; `main.log` "Model load size estimate ... 16384" lines are the only per-load marker, and the embedder's loads (context 2048) do not produce one.
4. LM Studio rotated its server log mid-run (10 MB cap); my own `/api/v0/models` poller (the only way to see loaded state without `lms`) inflates that log by about 8 KB per poll. Anyone repeating this should poll at 1 s or slower.
5. A few extra chat completions with 4 messages appear in the interleaved run (extractor repair re-prompts); unrelated to order.

## Feedback & recommendations

- State the K-070 claim precisely: the saving is per document (2 swaps instead of 2N per document), not per batch; a REST batch of N documents still swaps about 2N times in total (observed 15 switches, 8 qwen loads for 8 documents).
- K-102 needs its own forced reproduction if a live acceptance claim is wanted: ingest through MCP (per-job threads) into a throwaway workspace while LM Studio is one-model-at-a-time (e.g. after `lms unload --all`), or add a harness hook that delays the embedder.
- A host-level fact worth recording: this LM Studio configuration can keep several models resident, so "one model at a time" (SERVER.md section 1.12 wording) is a worst case, not an invariant.
- Re-run on a quiet LM Studio to confirm the conc wall-time effect.
