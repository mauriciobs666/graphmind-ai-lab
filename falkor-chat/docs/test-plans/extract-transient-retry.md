# Transient provider-error retry — live proof test plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-067

## 1. Scope & objective

Live, black-box proof of the K-067 retry mechanism (`falkorchat/retry.py`, wired into
`IngestionPipeline.extract_chunk` and `EmbeddingWorker.embed_chunk`) at HEAD `6519db83`, through
the real producer → store → consumer path: REST `POST /documents` and MCP `ingest_document` →
background jobs → FalkorDB `Document.status`/`pendingJobs` → `search_documents` /
`get_document`.

The plan has two halves (plan `falkor-chat/docs/plans/extract-transient-retry.md` §5 "Live proof
run"):

- **Half A — proxy legs (a)–(d′), incl. (c′): executed in this run.** Proves wiring, timing, the
  K-051 status contract, the liveness gate and rung 3b. Does **not** prove classifier fit (the
  injected bodies are copies of the allow-list's source strings, review M-3).
- **Half B — real-fault legs (e)–(g): specified only (§6), NOT executed in this run.**
  Stakeholder scope 2026-10-15: no `lms.exe unload`, `server stop`/`start`, or anything else that
  disrupts the shared LM Studio; these legs wait for a session with the stakeholder at the machine.
  §6 is written so that run can execute from this plan alone.

## 2. References

- Plan v3: `falkor-chat/docs/plans/extract-transient-retry.md` — §3.1 (classifier), §3.2 (policy,
  2/8/30 s × `[1, 1.5)` jitter), §3.2.1 (pre-stated (g) outcomes), §3.5 (gate), §3.6 (log-line
  contract), §5 "Live proof run" + "Acceptance mapping".
- Review: `falkor-chat/docs/reviews/extract-transient-retry.md` Pass 3 (P3-1, P3-2, N-3 nit).
- Ledger: `falkor-chat/docs/plans/extract-transient-retry-coordination.md` (U5).
- Code under test: `server/falkorchat/retry.py`, `ingestion.py` (label `extract chunkId=<id>`),
  `embedding.py` (label `embed chunkId=<id>`), `background.py` (`background chunk embed failed` /
  `background extract failed` tracebacks), `repository.py::chunk_document_is_processing`.
- Offline coverage already in place (not duplicated here): U-1…U-17, I-1…I-4 in `test_retry.py`,
  `test_transport.py`, `test_ingestion.py`, `test_embedding.py`, `test_background.py` (2988 passed).

## 3. Risk assessment (what this run exists to catch)

| Risk | Why offline tests can't see it | Leg(s) |
|---|---|---|
| R1 Retry wired in a test seam but not in the app-built objects (`app._build_default_app` passes no policy) | offline tests inject `sleep`/policy | all |
| R2 Real urllib transport surfaces a fault as a different `kind` than the fakes (e.g. reset → `network`, truncated body → `read`) | fakes raise prepared exceptions | (d), (d′) |
| R3 Real sleep spacing off (jitter direction, wrong delay index) | offline sleep is a recorder | (a) |
| R4 Gate Cypher wrong against a real graph / real delete | offline repo fakes | (c), (c′) |
| R5 Retry produces duplicate writes / orphans in the real graph | — | orphan query after every leg, (c′) |
| R6 MCP's concurrent daemon threads behave differently from REST's sequential chain | thread model differs | every leg × {REST, MCP} |
| R7 Classifier misses LM Studio's *real* fault shapes | bodies here are copies | Half B only (not run) |

## 4. Environment & data setup

- **FalkorDB**: the shared instance on `127.0.0.1:6379`. Writes go to **`ws:retryprobe` only**.
  Never write `ws:agent-team`, `ws:acme`, `ws:demo`, `reference`. Read-only `GRAPH.RO_QUERY` on
  `ws:agent-team` for the baseline counts.
- **Workspace**: `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh retryprobe` (DDL only; touches
  `reference` DDL only, never its data), then `./scripts/seed_demo.sh retryprobe` (writes only
  `ws:retryprobe`: User `u1`, Agent, demo channel/thread) so the configured actor resolves
  (`UnknownActorError` otherwise). Not `create_workspace.sh`.
- **Model config**: scratch copies of the two `ModelGateway` files — `opencode.json` (copied from
  `~/.config/opencode/opencode.local.json`, `lmstudio.options.baseURL` →
  `http://localhost:1299/v1`) and `models.json` (byte copy of `falkor-chat/config/models.json`).
  The shipped files are never edited.
- **Proxy**: stdlib `ThreadingHTTPServer` on `localhost:1299` forwarding to `localhost:1234`.
  Faults are *armed* per leg-run: `{nonce, path, script}`. A request is faulted only if its path
  matches **and** its body contains the armed nonce; the script is indexed by **per-body-hash
  hit count** (sha256 of the request body), so a retry (identical body) advances the script and
  each leg-run's nonce gives fresh counters (plan §5 Timing rule; no proxy reset). Logs one JSON
  line per request: `ts, path, sha256[:16], nonce_match, hit_n, action, upstream_status`.
  Fault actions: `400term` (`{"error":"terminated"}`), `500html`, `400ctx` (verbatim
  `exceed_context_size_error` body from `opencode/docs/test-reports/devops-opencode-headless-report.md`),
  `500` (always), `reset` (read request, close socket with no status line), `trunc` (`200`,
  `Content-Length: 4096`, write 10 bytes, close), `fwd`.
- **Server**: `uvicorn falkorchat.app:app --port 8311` direct (no `--reload`, no
  `start_server.sh`/`start_demo.sh`), env `FALKORCHAT_WS_ID=retryprobe`,
  `FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_WORKFLOW_ENABLED=0`, `FALKORCHAT_EMBEDDING_DIM=1024`,
  `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` → scratch copies, and
  `--log-config` → a scratch dictConfig that routes the root logger at **INFO** with timestamps
  to a file (the app configures no logging itself, so without this the §3.6 INFO "abandoning
  retry" line is not emitted anywhere).
- **Text**: every leg-run's text starts `Probe run <nonce>.` (`uuid4().hex[:8]`); in a
  multi-chunk document **every chunk** carries the nonce (review Pass 3 N-3 nit).
- **Scratch**: proxy, driver, per-leg logs, `nonce → documentId → body hashes` records under
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u5/`.

## 5. Half A test items (executed)

Common to every item (TP-001…TP-012):
- **Pre**: orphan query `MATCH (e:Entity) WHERE NOT ()-[:ABOUT]->(e) RETURN count(e)` on
  `ws:retryprobe` reads 0.
- **During**: a driver thread polls `Document.status`/`pendingJobs` every ~0.5 s and records
  every change (proves `processing` with the not-yet-decremented count inside the retry window).
- **Post**: final `status`/`pendingJobs`; orphan query = 0; server log lines joined by
  `chunkId=<id>` for both the extract and the embed job; proxy lines for the leg-run's nonce;
  **consumer half** — `search_documents` (REST `GET /documents/search` for REST runs, MCP
  `search_documents` for MCP runs) returns the document's chunk for `ready`; `get_document`
  returns it with `status: failed` for a `failed` one.
- Proxy gaps = `ts(hit n+1) − ts(hit n)` for the faulted body hash.

| ID | Leg | Transport | Script (path) | Expected | Pri | Type |
|---|---|---|---|---|---|---|
| TP-001 | (a) chat | REST | `/v1/chat/completions`: `400term`,`400term`,`500html`, then fwd | `ready`; entities present; WARNINGs `extract chunkId=…` attempts 1/4, 2/4, 3/4 with `kind=http status=400/400/500`; 4 chat hits; gaps ≥ 2 s, ≥ 8 s, ≥ 30 s (and < 3, < 12, < 45 s per jitter bound) | P1 | e2e |
| TP-002 | (a) chat | MCP | same | same | P1 | e2e |
| TP-003 | (a) embed | REST | `/v1/embeddings`: same script | `ready`; chunk embedding present; WARNINGs `embed chunkId=…` 1/4,2/4,3/4 `kind=http 400/400/500`; 4 embed hits; same gap bounds | P1 | e2e |
| TP-004 | (a) embed | MCP | same | same | P1 | e2e |
| TP-005 | (b) permanent | REST | chat: `400ctx` (always) | `failed` within ~1 s of the hit; exactly **1** chat hit; no retry WARNING for the extract label; `background extract failed` traceback | P1 | e2e |
| TP-006 | (b) permanent | MCP | same | same | P1 | e2e |
| TP-007 | (c) outage, 3-chunk | REST | chat **and** embeddings: `500` always | first job (embed chunk 1) hits 4×; `failed` 40–60 s after its first hit; every later job (5) makes exactly **1** hit and logs the gate INFO `abandoning retry`; record total drain time | P1 | e2e |
| TP-008 | (c′) gate during sleep | REST | chat: `500`,`500`, then fwd; embeddings fwd | delete via public REST `DELETE /documents/{id}` 3 s after chat hit 2; exactly **2** chat hits, never a 3rd; orphan count **0**; INFO `extract chunkId=… abandoning retry`; no attempt-3 WARNING (no `attempt 2/4` follow-up hit). Void-run rule: delete outside the window → repeat with a fresh nonce | P1 | e2e |
| TP-009 | (d) reset | REST | chat: `reset` ×1, then fwd | `kind=network`, attempt 1/4 WARNING, then `ready` | P1 | e2e |
| TP-010 | (d) reset | MCP | same | same | P2 | e2e |
| TP-011 | (d′) truncated body | REST | chat: `trunc` ×1, then fwd | `kind=read`, attempt 1/4 WARNING, then `ready` (rung 3b) | P1 | e2e |
| TP-012 | (d′) truncated body | MCP | same | same | P2 | e2e |

Notes:
- (c) is REST-only per plan; (c′) is REST because the delete timing is driven from the proxy log
  and the REST chain is the deterministic one. Both are the plan's own transport choice.
- **Spacing assertions (TP-001…TP-004) use 1-chunk documents only** (plan Timing rule).
- **Pass/fail is against the plan's stated values**; the jitter upper bounds are added from §3.2's
  table (`[1, 1.5)`).

## 6. Half B — real-fault legs (SPECIFIED, NOT RUN in this session)

Run direct to LM Studio (`baseURL` `http://localhost:1234/v1` in a scratch `opencode.json`), same
server wiring as §4 otherwise. Requires the stakeholder at the machine.

**Coordinator obligation (plan §5, review N-7), before (e):** `teco` either tells every in-flight
agent not to call `ingest_document` (or anything else reaching LM Studio) until QA reports the
window closed, or schedules U5-B when no other agent is in flight. QA does not start (e) until
the coordinator confirms one of the two.

**Pre-leg checks — all three, immediately before each of (e), (f), (f-12 s), (g), and any (g)
re-run.** Any miss → do not start; wait and re-check, or ask the coordinator.
1. `ws:agent-team`: `MATCH (d:Document {status:'processing'}) RETURN count(d)` = **0**; record
   `MATCH (d:Document {status:'failed'}) RETURN count(d)` as the failed baseline.
2. `pgrep -af '[r]un-sweep|[m]odel-bench'` returns nothing (a non-sweep match → coordinator's call).
3. LM Studio quiet check, **carried verbatim from review Pass 3 P3-2**:
   > - Specify `lms.exe log stream --source server --json`.
   > - Before trusting silence, require a positive control: one `curl localhost:1234/v1/models` must
   >   produce a `server.log` line in that stream.
   > - If the control does not appear, check 3 is not evaluable. Proceed only on checks 1–2 plus the
   >   coordinator's explicit confirmation, and record that in the report.
   > - Still unverified: whether a `POST` completion request logs under `server`. QA records that
   >   on the first real request.

   Concretely: start `/mnt/c/Users/mauri/.lmstudio/bin/lms.exe log stream --source server --json`
   ≥ 60 s before the leg, run the positive-control `curl -s localhost:1234/v1/models`, confirm a
   `{"type":"server.log",…}` line appears, then require **no request line for the last 60 s**.

**Post-leg re-check, after each leg and at teardown:** re-run check 1's counts; `processing` back
to 0 within a few minutes (a stuck one is reported); state the `failed` delta even when 0; list
each new failed `documentId`/`title` with the coinciding leg — the coordinator owns its re-drive;
QA never re-drives a `ws:agent-team` document. Restore `lms ps` state at the end (before = after).

**Human fallback** (only if an `lms.exe` call errors or hangs > 20 s): the stakeholder, in LM
Studio's Developer tab — (e) **Eject** the chat model on QA's "now"; (f) toggle **Status:
Running** off, wait, on. QA records the wall-clock action time.

| ID | Leg | Steps | Record / expected |
|---|---|---|---|
| TP-101 | (e) unload mid-generation | REST-ingest a 1-chunk ~1,000-char entity-dense doc; 1–2 s after the response, while `lms ps` shows the chat model, run `lms.exe unload qwen/qwen3-4b-2507` | what the in-flight request surfaced as (`kind`/`status`/`detail` prefix); whether the retry recovered after JIT reload; final status |
| TP-102 | (f) server bounce 5 s | same doc shape; while extract is in flight: `lms.exe server stop`, wait 5 s, `lms.exe server start` | `kind=connection` or `network` during the bounce, recovery within the 2/10/40 s cumulative schedule |
| TP-103 | (f) server bounce 12 s | as TP-102 with a 12 s wait (lands between attempts 3 and 4) | as TP-102 |
| TP-104 | (g) unproxied contention, MCP | `lms.exe unload --all`; MCP-ingest one 8-chunk document (16 concurrent jobs) | per job: first-attempt `kind`/`status`/`detail` prefix, attempts used, final status; document status; wall time |
| TP-105 | (g) unproxied contention, REST | repeat TP-104 via REST | same |

Pre-stated readings for (g) (plan §3.2.1), applied mechanically:
- any embed job with `kind=timeout` → apply §3.2.1 (flip `EMBED_CHUNK_RETRY.retry_timeouts`, U-17
  and the U-5b row updated and passing, (g) re-run once MCP then REST with fresh nonces);
- a transient-looking 4xx body matching no marker → add the marker per §3.2.1 and re-run;
- a raw non-`ProviderCallError` in the log → return to `architect`;
- `ready` → "converged in this run" (n = 1), never a convergence claim;
- `failed` with transient kinds in WARNING and give-up lines → mechanism delivered; K-067 closure
  rule applies (F-4 filed with (g)'s table before K-067 leaves `BACKLOG.md`; no "cured" claim in
  `HISTORY.md`).

**§3.2.1 condition 3 on the (g) re-run, carried verbatim from review Pass 3 P3-1:**
> - Define the set independently of the flip: every `background chunk embed failed` traceback
>   whose `ProviderCallError` message contains `TimeoutError` or `timed out`, **plus** every
>   `embed chunkId=…` line carrying `kind=timeout`.
> - Fail condition 3 if any job in that set lacks the WARNING, INFO, or give-up line.
> - Allow "not exercised" only when that independent set is empty.
> - Add a precondition that the server process started after the constant edit (process start
>   time later than the `retry.py` mtime).

Mechanically: `stat -c %Y server/falkorchat/retry.py` < the uvicorn PID's start time
(`ps -o lstart= -p <pid>`), checked and recorded before the re-run; at least one job in the set
must show the WARNING (plan §3.2.1 condition 3's "at least one" clause).

**Teardown (both halves):** stop uvicorn and the proxy; `GRAPH.DELETE ws:retryprobe` (if a
permission check blocks it, do not work around it — record the exact command for the
stakeholder); final `ws:agent-team` re-check and failed delta; shared `opencode.json` untouched;
(Half B) restore loaded-model state.

## 7. Entry / exit criteria

- **Entry (Half A)**: HEAD `6519db83`; FalkorDB answers PONG; `localhost:1234/v1/models` answers;
  ports 1299 and 8311 free; `ws:retryprobe` absent or empty; `ws:agent-team` baseline recorded.
- **Exit (Half A)**: every TP-001…TP-012 scored pass/fail/blocked with evidence; orphan count 0
  after every leg (or reported); teardown done or its remainder recorded.
- **Exit (Half B)**: not in this run.

## 8. Out of scope

- Half B execution (stakeholder scope 2026-10-15).
- The falkor-chat full pytest suite and `./scripts/test_queries.sh` (ledger rule; offline suite
  already green at integration).
- Classifier fit to LM Studio's real fault bodies (Half B's job; Half A cannot prove it).
- `ingest_documents` (batch) chain timing, the per-request breaker (F-3), timeouts (`kind=timeout`
  is non-retried by design and not injectable without a 30–180 s hang).
