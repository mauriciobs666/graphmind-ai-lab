# Transient provider-error retry — live proof report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-067

## 1. Summary

- **What was tested:** Half A (proxy legs (a)–(d′), incl. (c′)) of the live proof in
  `falkor-chat/docs/test-plans/extract-transient-retry.md`, which carries plan
  `falkor-chat/docs/plans/extract-transient-retry.md` v3 §5. This was the first end-to-end run of
  the real producer → store → consumer path for the K-067 retry: REST `POST /documents` and MCP
  `ingest_document` → background extract/embed jobs → `ws:retryprobe` → `search_documents` /
  `get_document`.
- **When / against what:** 2026-10-15, 19:01–19:12 UTC, HEAD `6519db83`.
  - The server was uvicorn PID 69080 on `127.0.0.1:8311`, `--reload` off. It started at
    16:01:08 -03:00, after `retry.py`'s mtime of 14:40:22 -03:00.
- **Verdict: PASS. All 12 Half A items passed and no K-067 defect was found.**
- **Half B, legs (e)–(g), was not run.** Per the stakeholder's 2026-10-15 scope, it is specified
  in test plan §6, with review P3-1/P3-2 carried verbatim, and is waiting for a session with the
  stakeholder at the machine.
- **Out-of-scope finding:** one product defect unrelated to K-067 (§4, D-1).
- **Testability finding:** one item that affects Half B (§6, F-1).
- CPG: considered, not relevant — black-box live run against the running server; the code-level
  test-gap analysis was done at the U4a–U4c step gates on `cpg_falkorchat`, and this run's
  question (does the wired app behave as specified on real I/O) is not a graph query.

## 2. Environment as run

- **FalkorDB:** the shared `127.0.0.1:6379`. All writes went to `ws:retryprobe` only.
  - `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh retryprobe` was run, then
    `./scripts/seed_demo.sh retryprobe`. The seed writes only `ws:retryprobe`; it is needed so that
    actor `u1` resolves.
- **Model config:** scratch copies, with the shipped files untouched.
  - `opencode.proxy.json` is a copy of `~/.config/opencode/opencode.local.json` with
    `lmstudio.options.baseURL` set to `http://localhost:1299/v1`.
  - `models.json` is a byte copy of `falkor-chat/config/models.json` (confirmed with `cmp`).
- **Server env:** `FALKORCHAT_WS_ID=retryprobe`, `FALKORCHAT_ENABLE_AGENT=1`,
  `FALKORCHAT_WORKFLOW_ENABLED=0`, `FALKORCHAT_EMBEDDING_DIM=1024`.
  - `--log-config` pointed at a scratch dictConfig that sends the root logger at INFO to
    `server.log`. This was needed because of F-1.
- **Proxy:** the stdlib `ThreadingHTTPServer` on `127.0.0.1:1299` → `127.0.0.1:1234`.
  - Faults are armed per leg-run with `{nonce, paths, script}`. A request is faulted only when
    its path matches **and** its body contains the nonce.
  - The script is indexed by the per-(path, body-sha256) hit count.
  - Each request gets one JSON log line with its arrival time (`t_arr`), path, `sha[:16]`, hit
    number and action.
  - Every chunk of a multi-chunk document carries the nonce (review Pass 3 N-3 nit).
- **Proxy restart:** the proxy was restarted once, after TP-005, to add the arrival timestamp.
  Counters are keyed per nonce, so no scored run spanned the restart.
- **Void run (not scored):** TP-005, first attempt, nonce `d4bea94d`.
  - The driver crashed in its own post-run query (FalkorDB rejects an `ORDER BY` after an
    aggregation that does not project the sort key). That is a harness defect.
  - On the product side the run had reached `failed` with `pendingJobs` 0 and one `400ctx` hit.
  - It was re-run with a fresh nonce.
- **LM Studio load:** across the whole run the proxy saw 61 requests. 28 were forwarded
  `POST`s, 3 were forwarded `GET /v1/models`, and the remaining 30 were faults the proxy served
  itself, which LM Studio never saw.
- **Scratch evidence**, all under
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u5/`:
  - `proxy.py`, `driver.py`, `commands.log` (every driver invocation);
  - `runs/TP-0NN.json` and `.out` (per leg-run: timeline, proxy lines, server lines, graph state);
  - `proxy.log.final.jsonl`, `server.log`, `records.md`.

## 3. Results

Expected values come from plan §5. For the gap figures I added the upper bounds from §3.2's
jitter table: gap 1 ∈ [2, 3) s, gap 2 ∈ [8, 12) s, gap 3 ∈ [30, 45) s.

- **Proxy gaps** are measured between request arrivals for the faulted body hash. Every injected
  fault answers at once, so each gap is the backoff sleep plus milliseconds.
- **State (in window)** is the driver's 0.5 s poll of `Document.status`/`pendingJobs` during the
  retry window. It shows `processing` with the not-yet-decremented count.
- **Orphans** is `MATCH (e:Entity) WHERE NOT ()-[:ABOUT]->(e) RETURN count(e)` on
  `ws:retryprobe`, read before → after each leg.

| ID | Leg | Via | Faulted hits / gaps | Retry log (by `chunkId`) | State (in window) → final | Orphans | Consumer half | Result |
|---|---|---|---|---|---|---|---|---|
| TP-001 | (a) chat | REST | chat ×4; 2.936 / 11.227 / 30.599 s | extract 1/4, 2/4, 3/4; `kind=http` 400/400/500 | `processing`/1 → `ready`/0 at 54.81 s; 11 `ABOUT` | 0→0 | REST search returns the chunk | **pass** |
| TP-002 | (a) chat | MCP | chat ×4; 2.834 / 8.414 / 35.078 s | extract 1/4, 2/4, 3/4; 400/400/500 | `processing`/1 → `ready`/0 at 57.12 s; 14 `ABOUT` | 0→0 | MCP search returns the chunk | **pass** |
| TP-003 | (a) embed | REST | embed ×4; 2.574 / 11.591 / 38.935 s | embed 1/4, 2/4, 3/4; 400/400/500 | `processing`/2 → `ready`/0 at 64.4 s; embedding present | 0→0 | REST search returns the chunk | **pass** |
| TP-004 | (a) embed | MCP | embed ×4; 2.629 / 9.985 / 37.131 s | embed 1/4, 2/4, 3/4; 400/400/500 | `processing`/2, then /1 from 10.37 s (extract done) → `ready`/0 at 52.13 s; embedding present | 0→0 | MCP search returns the chunk | **pass** |
| TP-005 | (b) permanent | REST | chat ×1 (`400ctx`) | none; one `background extract failed` traceback | `failed`/0 at 0.51 s (chat hit at 0.281 s) | 0→0 | REST `get_document`: `failed`, text intact | **pass** |
| TP-006 | (b) permanent | MCP | chat ×1 (`400ctx`) | none; one `background extract failed` | `failed`/1 at 0.39 s → `failed`/0 at 0.89 s | 0→0 | MCP `get_document`: `failed`, text intact | **pass** |
| TP-007 | (c) outage, 3 chunks | REST | first job (embed c1) ×4, gaps 2.476 / 11.114 / 32.213 s; the 5 later jobs ×1 each | embed c1 1/4, 2/4, 3/4 + `giving up after 4 attempts`; each later job: gate INFO `abandoning retry` + its traceback | `processing`/6 → `failed`/0 at 46.35 s | 0→0 | REST `get_document`: `failed` | **pass** |
| TP-008 | (c′) gate during sleep | REST | chat ×**2**, gap 2.548 s; no 3rd hit in the 20 s watch after settle, nor anywhere later in the proxy log | extract 1/4, 2/4 (`retrying in 10.3s`); post-sleep INFO `abandoning retry`; no attempt-3 WARNING | `processing`/1 → deleted | **0→0** | n/a (deleted; `DELETE` → 200) | **pass** |
| TP-009 | (d) reset | REST | chat ×2; 2.473 s | extract 1/4 `kind=network status=None` (`RemoteDisconnected`) | `processing`/1 → `ready`/0 at 12.06 s; 10 `ABOUT` | 0→0 | REST search returns the chunk | **pass** |
| TP-010 | (d) reset | MCP | chat ×2; 2.818 s | extract 1/4 `kind=network` | `processing`/1 → `ready`/0 at 13.39 s; 14 `ABOUT` | 0→0 | MCP search returns the chunk | **pass** |
| TP-011 | (d′) truncated body | REST | chat ×2; 2.085 s | extract 1/4 `kind=read status=None`, `reading response failed: IncompleteRead` | `processing`/1 → `ready`/0 at 11.55 s; 12 `ABOUT` | 0→0 | REST search returns the chunk | **pass** |
| TP-012 | (d′) truncated body | MCP | chat ×2; 2.067 s | extract 1/4 `kind=read`, `IncompleteRead` | `processing`/1 → `ready`/0 at 11.88 s; 12 `ABOUT` | 0→0 | MCP search returns the chunk | **pass** |

Times are seconds after the ingest call. `ABOUT` is the number of `ABOUT` edges on the
document's chunk.

### 3.1 Per-leg evidence

**TP-001 (a) chat, REST.** Server log for `chunkId=020859a6135441bbaa1eb11cb950200f`:

```
16:03:22.965 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=020859a6135441bbaa1eb11cb950200f, kind=http, status=400), attempt 1/4, retrying in 2.9s: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1299/v1/chat/completions: HTTP 400 Bad Request: {"error":"terminated"}
16:03:25.902 WARNING … (extract chunkId=020859a6…, kind=http, status=400), attempt 2/4, retrying in 11.2s: … {"error":"terminated"}
16:03:37.128 WARNING … (extract chunkId=020859a6…, kind=http, status=500), attempt 3/4, retrying in 30.6s: … HTTP 500 Internal Server Error: <!DOCTYPE html>…
```

- The embed job for the same chunk was forwarded and succeeded. A successful job logs nothing,
  so the server log has no embed line for this `chunkId`.
- The same holds for every (a)-chat, (d) and (d′) run.
- TP-002 logged the same three lines on thread `Thread-5 (_safe_extract)`, the MCP daemon
  thread: `retrying in 2.8s`, `8.4s`, `35.1s`.

**TP-003 / TP-004 (a) embed.**
- The lines are the same with the label `embed chunkId=…` and
  `lmstudio/text-embedding-qwen3-embedding-0.6b @ …/v1/embeddings`.
- In TP-003 (REST), the extract job's single chat hit arrived at **54.023 s**, after the embed
  job's retries ended. This is direct live confirmation of plan §3.5: on the REST path a
  sleeping retry delays the rest of that request's sequential `BackgroundTasks` chain.
- In TP-004 (MCP), the chat hit arrived at 0.323 s, concurrent with the embed's first attempt.
  That matches the MCP thread model.

**TP-007 (c) outage.** The give-up and the gate cascade, all within 19 ms:

```
16:10:00.499 WARNING falkorchat.retry giving up after 4 attempts (embed chunkId=7c2f8ab9cfc04de8a00f52b09458b9d5, kind=http, status=500): …
16:10:00.499 ERROR falkorchat.background background chunk embed failed (chunkId=7c2f8ab9…)
16:10:00.503 INFO  falkorchat.retry extract chunkId=7c2f8ab9…: owning document no longer processing; abandoning retry (kind=http, status=500)
16:10:00.508 INFO  falkorchat.retry embed chunkId=ab227d21752b4543bb7a2988f1ff975c: owning document no longer processing; abandoning retry (kind=http, status=500)
16:10:00.511 INFO  falkorchat.retry extract chunkId=ab227d21…: … abandoning retry …
16:10:00.515 INFO  falkorchat.retry embed chunkId=4830f2b619cb46c88ed8e7e2bfc90bbd: … abandoning retry …
16:10:00.518 INFO  falkorchat.retry extract chunkId=4830f2b6…: … abandoning retry …
```

Each INFO line is followed by that job's own `background … failed` traceback.

- **Time to `failed`:** the first job's first hit was at 0.095 s and its 4th hit at 45.898 s. The
  give-up came 45.8 s after hit 1, inside the plan's 40–60 s range.
- **Total drain time:** 46.35 s from ingest. The last of the 5 single hits arrived at 45.919 s.
- **Final chunk state:** all 3 chunks have no embedding and 0 `ABOUT` edges.

**TP-008 (c′) gate during sleep.** Chunk `5aab5167fb534f59b42dc6419565b8f7`.

| Time / offset | Event |
|---|---|
| 0.297 s | chat hit 1 (`500`) |
| 2.846 s | chat hit 2 (`500`) |
| 16:10:18.562 | WARNING `attempt 2/4, retrying in 10.3s`, logged after the pre-sleep gate read returned true |
| hit 2 + 3.000 s (16:10:21.570) | driver `DELETE /documents/bea78aad…` → `200 {"deleted": true}` |
| 16:10:28.876 | INFO `extract chunkId=5aab5167…: owning document no longer processing; abandoning retry (kind=http, status=500)` |
| 16:10:28.877 | ERROR `background extract failed` traceback |

- The INFO line came 10.3 s after the WARNING, so it was the **post-sleep** read that tripped.
- **The run is valid under the void-run rule.** The delete landed after the pre-sleep read and
  7.3 s before the sleep ended.
- The proxy log holds exactly 3 request lines for nonce `9d1f3b94`: embed ×1 (forwarded) and
  chat ×2. There is no 3rd chat hit.
- Orphans were 0 before and 0 after.

**TP-005 / TP-006 (b).** Each run has a single traceback whose last line is:

```
falkorchat.transport.ProviderCallError: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1299/v1/chat/completions: HTTP 400 Bad Request: {"error":{"code":400,"message":"request (11381 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error",…}}
```

Neither run logged a `falkorchat.retry` line.

**TP-009 … TP-012 (d)/(d′).** The attempt-1 messages, verbatim prefixes:
- (d): `…/v1/chat/completions: RemoteDisconnected: Remote end closed connection wi…` (rung 3,
  `kind=network`).
- (d′): `…: reading response failed: IncompleteRead: IncompleteRea…` (rung 3b, `kind=read`). This
  is the first live exercise of rung 3b.

### 3.2 Nonce → document → body hash (plan §5 record)

| ID | Via | Nonce | documentId | chunk(s) | body hashes (sha256[:16]) × hits |
|---|---|---|---|---|---|
| TP-001 | rest | `78d69d49` | `569b39f7f4df45929bb249f90b30f15e` | `020859a6…` | embeddings `01bb264d20d07773`×1; chat `05870d77691742d2`×4 |
| TP-002 | mcp | `f271b5a7` | `038dab51e3154286a9a09b491b5ba21f` | `2ef26d75…` | chat `f3f82044930d9742`×4; embeddings `bc27bcf5c86bd79f`×1 |
| TP-003 | rest | `5114600c` | `51ce8613d0ac44d1a4915bc35261466a` | `672b4b06…` | embeddings `ea6a8f4c0d65c78c`×4; chat `f41fad9b71338b3f`×1 |
| TP-004 | mcp | `a9fabb5c` | `f0bf66d5456c4311a9857169d724adfb` | `793d4300…` | embeddings `cf2d38ae58cbacf0`×4; chat `34d1e4e363d16735`×1 |
| TP-005 | rest | `70f6ea47` | `e72a2ddbc1de42c1ad074008c25fc7b7` | `47d78c82…` | embeddings `f03f89b41cb563dc`×1; chat `c3b7c78ce59cad05`×1 |
| TP-006 | mcp | `1bbf10d1` | `d2d31e1f1ebf47e7a698784544885b06` | `2123b615…` | chat `e3b0d61623e134e5`×1; embeddings `be9c9980ffb24058`×1 |
| TP-007 | rest | `17bab298` | `51b1d6097c29495d918e3949e175c5b1` | `7c2f8ab9…`, `ab227d21…`, `4830f2b6…` | embeddings `844eb1176ecbc2fa`×4; chat `6d5ffb1bc3f816c4`×1; embeddings `3ecb11c89808ad76`×1; chat `f414227db5b12d56`×1; embeddings `c27f45f4350fae76`×1; chat `092fd6759d3591ca`×1 |
| TP-008 | rest | `9d1f3b94` | `bea78aad2b0a4a7e8288e0117505a1b7` | `5aab5167…` | embeddings `d9c573cf1c1dba46`×1; chat `78a1538bc2785885`×2 |
| TP-009 | rest | `9e6bad6c` | `6a1b7485513b4d20bd1567f130b39b25` | `a53c5200…` | embeddings `dbc972401634222d`×1; chat `931c7635a4bdc20d`×2 |
| TP-010 | mcp | `a6049550` | `74f293a6b4824cbbbc2efa35149d9e5c` | `8123951c…` | chat `be848c46b94c2834`×2; embeddings `09b146f6bf5dddf7`×1 |
| TP-011 | rest | `73a74f66` | `1a857a85a9d44545a44f2cc7c9b78953` | `928e463d…` | embeddings `32ba0bda3d711664`×1; chat `8d01f1e3f87ddc1a`×2 |
| TP-012 | mcp | `511df1b2` | `63de6607735b4a84a1d973ae7235b996` | `43c32f14…` | chat `fead918223c260ff`×2; embeddings `b2f277482863a2c6`×1 |

Void run: `d4bea94d` → `43af56b07c9a48e18aea55bb3f3ebe96`, embeddings `2c4c1a223725ca54`×1, chat
`a8eabdfdb0cfb85f`×1.

### 3.3 Shared-state checks

- **`ws:agent-team`** (read-only), checked at start, after the last leg and after teardown:
  - `ready` 340, `failed` 12, `processing` 0 at every check;
  - the **failed delta was 0**.
  - The brief quoted 339 `ready`; the 340 here is the count I read at the start of this run.
  - This server never reached LM Studio's direct port, and `ws:agent-team`'s server (port 8200)
    never went through the proxy.
- **`ws:retryprobe` before teardown:** 8 `ready` and 4 `failed`. The 4 failed are the void run,
  TP-005, TP-006 and TP-007; TP-008's document was deleted. Workspace-wide orphan count: 0.

## 4. Defects

**No K-067 defect was found.** One incidental defect surfaced, outside K-067's scope:

**D-1 — `fusion._fuzzy_query` sends an unescaped hyphenated token to RediSearch, so suggested-tier
fusion fails for entities like `Kestrel-0`** (severity: low; out of scope for K-067).
- **Steps:** ingest any document whose extracted entity name contains `-<digit>`. In this run
  that was `Kestrel-0` and `Okonkwo-<nonce>0`.
- **Expected:** fuzzy candidate lookup runs.
- **Actual:** `ERROR falkorchat.background background fuse failed (entityId=…)` with
  `redis.exceptions.ResponseError: RediSearch: Syntax error at offset 8 near -0`. The same error
  also appeared `near Kestrel` and `near Norbre`.
  - There were 6 occurrences, all in `server.log`, for example at 16:04:17.175.
  - `_fuzzy_query` builds `%tok%` per whitespace token with no escaping
    (`server/falkorchat/fusion.py:34-42`).
- **Impact:** `_safe_fuse` isolates the failure per entity. Entity creation, `ABOUT` edges and
  `Document.status` are unaffected; only the suggested-tier `SAME_AS` candidates for that entity
  are skipped.
- **Related:** this is the "same root pattern" that `BACKLOG.md` K-064 names as a related,
  unaddressed instance in `fusion._fuzzy_query`. K-064's own subject is implicit-AND recall; this
  is a syntax-error crash on punctuation.
- **Recommendation:** route to `coder`/`tdd-engineer` as its own item (escape RediSearch special
  characters per token). I did not fix it.

## 5. Coverage & gaps

**Covered live:**
- the wiring of both shipped policies into the app-built objects (R1);
- the real urllib ladder for 400, 500, `RemoteDisconnected` and `IncompleteRead`, i.e. rungs 1,
  3 and 3b (R2);
- real sleep spacing with upward jitter, all gaps inside their bounds (R3);
- the gate read against the real graph, both the pre-sleep read (TP-007's later jobs) and the
  post-sleep read (TP-008), with a real public-surface delete (R4);
- zero orphans after every leg (R5);
- REST and MCP thread models for (a), (b), (d) and (d′) (R6).

**Not covered:**
- **Half B (R7) — classifier fit to LM Studio's real fault shapes.** Half A cannot prove it
  (review M-3). It needs Half B.
- (c) and (c′) over MCP. The plan specifies REST for both, which is deliberate.
- `ingest_documents` batch chains.
- `kind=timeout`. It is non-retried by design; injecting one needs a hang of 30–180 s.

**Residual risk:** a sustained real outage on the MCP path, where every job fails together and
the jobs race to the gate, was exercised only offline. TP-007 is REST.

**Acceptance mapping (plan §5):**
- "transient retried with bounded backoff": live (a), (d) and (d′) **evidenced**.
- "bad chunk fails fast": live (b) **evidenced**.
- "exhausted budget → `failed`": live (c) **evidenced**.
- "idempotent on chunk + document state": orphan query after every leg, and (c′), **evidenced**.
- "classifier fires on real LM Studio faults": **open**, pending Half B.

## 6. Feedback & recommendations

**F-1 (testability; affects Half B) — the app configures no logging.**
- **Effect:** under uvicorn's default log config, every `falkorchat.*` INFO line is dropped. That
  includes the gate's `abandoning retry` line, which (c′) and P3-1 depend on. WARNINGs reach
  stderr only through Python's `lastResort` handler, bare and without a timestamp.
- **Verified:** I applied `uvicorn.config.LOGGING_CONFIG` via `dictConfig`, then logged from
  `falkorchat.retry`. The INFO line was not printed; the WARNING was printed bare; the root logger
  stayed at level 30 with no handlers.
- **What this run did:** it passed `--log-config` with a root-at-INFO dictConfig.
- **Recommendations:**
  - Half B must do the same. The test plan §4 wiring already includes it.
  - The plan's "capture the server log at INFO" should name the mechanism.
  - Longer term, the server (`start_server.sh` / `start_agent_team.sh`) should ship a log config.
    Without one, `ws:agent-team`'s 8200 process cannot show a gate abandonment at all.
- **Learning capture:** this would normally be filed to `ws:agent-team` as a learning-capture
  entry, but this unit's brief forbids writing `ws:agent-team`, so it is left for `teco` to file.
- **Proxy logging must use arrival time.** Forwarded requests complete seconds after they arrive,
  so a log timestamp taken at completion skews the last gap of (a). Already fixed in this run's
  proxy. Any re-run must keep `t_arr`.
- **Nit (N-3) handled:** every chunk of the multi-chunk document carried the nonce, so TP-007's
  six bodies are all distinct.

## 7. Teardown & stakeholder actions

**Done:**
- Server PID 69080 stopped; its log shows `Finished server process [69080]`.
- Proxy stopped; ports 1299 and 8311 are free.
- `redis-cli GRAPH.DELETE ws:retryprobe` → `OK`. `GRAPH.LIST` no longer shows it.
- The shared `opencode.json` and the shipped `config/models.json` are untouched.
- LM Studio's model state was not touched, beyond ordinary JIT use through plain requests.

**Left for the stakeholder:** Half B, legs (e)–(g), TP-101…TP-105. It needs:
- the stakeholder at the machine;
- `teco`'s coordinator confirmation (test plan §6);
- the P3-2 positive control on `lms.exe log stream --source server --json`.

No permission denials occurred in this run.
