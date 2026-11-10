# RCA: S10 ingest of 36 KB units into ws:pwq-test — 18 failed / 3 stuck processing

> **Status:** active · **Owner:** `analyst` · **Tracks:** python-web-quirks-kb (S10, S13)

Observations from 2026-11-10 09:20-09:24 -03. Read-only investigation. Inferences are marked **[inferred]**.

## 1. Symptom and impact

36 `POST /documents` to :8210 returned 201 (09:10:32-09:10:36, ~0.23 s apart, `pwq-test-8210.log:1146-1160`). After a 10-minute gate: 18 unit documents `failed` (all chunks embedded except u02 6/7), 3 `processing` with unembedded chunks (u21, u26, u29), 14 clean. Impact: S10 gate fails; if repeated on :8200 in S13 the stakeholder must DELETE and re-ingest each failed document (no `replace` in production, plan D9).

## 2. Evidence

- Server pid 1815524, up 51 min at check, never restarted since before the ingest; 9 threads, 0.2% CPU, no restart or reload.
- Log: 0 "background chunk embed failed", 25 "background extract failed", and all 25 are `ProviderCallError ... lmstudio/qwen/qwen3-4b-2507 @ .../chat/completions: TimeoutError: timed out`. First wave at 09:13:46-09:13:49 = POST time + ~180 s. Log last written 09:16:58.
- `falkor-chat/config/models.json`: `timeouts.extraction = 180`, `embedding = 30`; extraction and embedding models both LM Studio (`qwen/qwen3-4b-2507`, `text-embedding-qwen3-embedding-0.6b`).
- Graph `ws:pwq-test`, 09:22: 35 failed docs (18 ours + 17 prod-inherited) / 479 ready / 4 processing, then 3, then 2, with `pendingJobs` shrinking (u21 chunks unembedded 3 -> 1, u26 4 -> 2; one document went `processing` -> `ready` between two of my queries). So the stuck documents were still progressing, not dead.
- Probe: a 3-token chat completion to `qwen/qwen3-4b-2507` took 4.7 s wall (via curl, read-only). Host free RAM 614-669 MB, swap 1.4 GB used. LM Studio `/v1/models` answers.

## 3. Causal chain

**Q1 - why `failed` with all chunks embedded.** `report_document_job_done` (`repository.py:1155-1188`) sets `status='failed'` the instant any single job reports failure; first terminal write wins. Jobs are one embed + one extract per chunk (`background.py:_schedule_chunk_processing`, ~215-230). The embeds all succeeded; one extract timed out (180 s, `transport.py:157-166` raises `kind="timeout"`). `retry.is_transient` treats `timeout` as transient only if `retry_timeouts` is set, and it is `False` in both shipped policies (`retry.py:25-30`, `:122-123`), so a timeout is final with no retry. One timed-out extract therefore fails the whole document even though embedding is complete. `status` is a processing-outcome flag, not a "searchable" flag.

**Q2 - why chunks stay unembedded / `processing`.** REST schedules via Starlette `BackgroundTasks` (`api.py:165-190`): each request's tasks run **sequentially** (embed c1, extract c1, embed c2, extract c2, ...) on one thread; different documents run in parallel (thread limit 100, `config.py:~265`). Each extract blocks up to 180 s on timeout. A document's embed for chunk N therefore cannot start until extracts 1..N-1 returned or timed out. u26 (10 chunks) needs up to ~10 x 180 s = 30 min in the worst case. This explains `processing` with partial embeds: **still running, not dead, not lost** (no restart, counters falling). Failed documents' remaining jobs also continue: `_safe_embed_chunk` has no liveness gate, and the K-067 gate (`retry.py`, `repo.chunk_document_is_processing`) only stops *retries*, not the first attempt of later extracts - so they keep loading LM Studio after the document is already `failed`. The 15 unembedded chunks in failed docs (graph, 09:23) are mostly the 17 prod-inherited failures plus u02's one; some may still be pending **[inferred - not split by origin]**.

**Q3 - load or config.** Load-induced, with a thin config margin. 36 documents x first-chunk extract = 36 concurrent chat calls to one 4B local model whose host is memory-starved (swap in use). LM Studio queues them; the later ones exceeded 180 s in queue plus generation. The 4.7 s for a 3-token reply while the queue was likely still draining suggests slow serving **[inferred]**. No model/config defect was found: the model is the documented default, extraction timeout is the shipped 180 s. Ruled out: server restart/reload (pid age, no reload), embed provider failure (0 embed errors), lost in-process queue (counters still moving), dead task (no).

**Q4 - production :8200 / S13.** Same code path, same `config/models.json` (shared), same default thread limit 100, `FALKORCHAT_ENABLE_AGENT=1` (env of pid 1268, up 4 days, so it also serves older code **[inferred: not diffed]**). The same risk applies: back-to-back POSTs (this tool does ~4/s) fire all first extracts at once. Worse in production: `failed` is not self-healing and there is no `replace`. Also, `SERVER.md` documents that a `--reload`/restart mid-retry leaves documents `processing` for good - S13 must not write inside `falkor-chat/` or restart :8200 during the pass.

**Retrievability.** `repository.search_chunks` (`repository.py:~1266-1290`) filters only `seed.documentCurrent = true` (and the vector index requires an embedding); it does **not** filter on `Document.status`. A `failed` document whose chunks are embedded is returned by `search_documents` (read from code; no live search run). Only chunks lacking an embedding are invisible. `failed` therefore costs only the missing entity extraction (GraphRAG expansion) and the tool's own `gate`/`FAILED` handling (`pwq_kb.py:774-791`, `:869`), not vector search.

## 4. Root cause

Confidence: **confirmed** for the mechanism (code read plus log plus graph agree); **inferred** for the capacity numbers.

- **Root cause:** a document's terminal status is hard-failed by one extract provider timeout, timeouts are never retried, and the tool fires all documents at once, so the single local LM Studio model queues past the 180 s per-call timeout.
- **Trigger:** 36 near-simultaneous ingests (~8 s total) with low host RAM.
- **Contributing:** per-request sequential task chain (an embed waits behind earlier extract timeouts); no first-attempt liveness gate on queued jobs; `pwq_kb.py ingest` has no pacing/wait; `gate` waits a fixed 10 min, shorter than the worst-case 30 min chain; the test copy inherited 17 failed + 1 processing prod documents, which inflate the failed count and the unembedded totals of any naive query.

## 5. Severity and suggested fix

Severity: **major** for S13 (silent, non-self-healing failures on production; recoverable by DELETE + re-ingest, no data loss). Not a blocker for search quality (failed docs with embedded chunks remain searchable).

1. **Pacing in the tool (owner: `tdd-engineer`, `claude/cobb/scripts/pwq_kb.py`).** After each POST, poll the document until `status != processing` (and all chunks embedded) before posting the next; or cap concurrency at 1-2 documents. Test first: a fake transport that reports `processing` for N polls asserts the next POST is not sent; a `failed` outcome stops the run and reports the unit. Make `gate` poll to terminal state with a per-document timeout instead of a fixed 10 min.
2. **Treat `failed` with all chunks embedded explicitly in the tool (owner: `tdd-engineer`).** Report it as "searchable, entities missing" rather than a hard gate failure, if extraction is not an S10/S13 acceptance requirement - decision for `architect`/stakeholder (open question below).
3. **Server-side robustness (owner: `coder`/`tdd-engineer`, falkor-chat; not required for S13):** gate first-attempt extract on `chunk_document_is_processing` so queued jobs stop loading the model after a document fails; consider `retry_timeouts` for extraction (flip is pre-stated in `retry.py` plan §3.2.1). `devops` owns only env: a larger `timeouts.extraction` in `config/models.json` is a config choice, but raising it only delays the same failure under the same load, and :8200 shares the file with the live agent-team - do not change it casually.
4. **Environment (owner: `devops`):** free host RAM before S13 (swap is in use, ~600 MB free); consider smaller concurrent footprint in LM Studio. Do not restart :8200 mid-ingest.
5. **Prevention test:** an integration test (fake slow provider) that ingests >=10 documents and asserts the driver never has more than K documents in `processing`.

## 6. Open questions

- Is extraction (entities) part of S10/S13 acceptance, or only embeddings/search? Determines whether `failed`-with-embeds is a gate failure or a warning (fix 2).
- I did not run a live `search_documents` against a failed document and did not read the sealed directory; retrievability is from code only.
- Final state of u21/u26/u29 after all queued jobs drain was not observed (still moving at 09:23); re-run `gate` after the server goes quiet, filtering out prod-inherited documents, before concluding they end `failed` or `ready`.
