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
- **Half B, legs (e)–(g), ran on 2026-10-16** (section "Half B", below). **Verdict: PASS, 5 of 5
  after the D-2 fix.** On the first run it was 4 of 5. TP-101 re-run on `21ff9f9b` passed and D-2 is
  closed (B.7).
  - TP-101 fails, D-2: LM Studio's real mid-request unload body
    `{"error":"Model unloaded by user or API request."}` is a 400 that matches no marker, so it is
    not retried.
  - The (g) reading triggers no code change, and both (g) runs "converged in this run" (n = 1).
  - The Half A content below is unchanged from 2026-10-15.
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

## Half B — real-fault legs (e)–(g), TP-101…TP-105

> Run 2026-10-16, 08:45–09:10 -03:00, HEAD `6c8b41d1` (K-067 code from
> `91b65fed`/`128941a6`/`6519db83`), direct to LM Studio (`http://localhost:1234/v1`), per test
> plan §6.

**Half B verdict: PASS, 5 of 5 after the D-2 fix.** On the first run, TP-101 failed (D-2) and
TP-102…TP-105 passed. TP-101 re-run on 2026-10-16 against `21ff9f9b`: **pass**, so **D-2 is closed**
(B.7). The first-run detail is kept below as recorded.

First run (2026-10-16, 08:45–09:10):
- **TP-101 fails (D-2).** When the chat model is unloaded mid-generation, LM Studio returns
  `HTTP 400 {"error":"Model unloaded by user or API request."}`. No marker matches that body, so
  the classifier treats it as permanent and the document fails on attempt 1 with no retry.
- **TP-102 and TP-103 pass.** A server stop surfaces as `kind=network` (in-flight request) and
  then `kind=connection`; both are retried, and both runs recovered on attempt 3.
- **TP-104 and TP-105 pass, both `ready`, "converged in this run" (n = 1 each).**
  - TP-104 (MCP): the real cold-load fault was `HTTP 500` with an HTML body on 7 of 8 extract
    jobs. Each was retried and recovered on attempt 2.
  - TP-105 (REST): no faults at all.
- **(g) reading (B.4):** no pre-stated code change is triggered. No embed `kind=timeout`, no
  unmatched 4xx in (g), no raw non-`ProviderCallError`.
- **Open coordinator decision:** whether the §3.2.1 marker pre-authorization covers D-2's body
  (captured in (e), not (g)).

CPG: considered, not relevant — black-box live fault-injection run against the running server and
the real LM Studio; the question is what LM Studio returns and how the wired app classifies it, not
a graph query (code-level gap analysis was done at the U4 gates on `cpg_falkorchat`).

### B.1 Environment and pre-run state

- Coordinator confirmation (§6, N-7): `teco` confirmed at dispatch that no other coordinated
  agent is in flight and none will be dispatched until this unit returns.
- `ws:agent-team` baseline at 08:45 -03:00: `processing` 0, `failed` **14**, `ready` 340. The 14
  include `4bcc13a3…` and `0c765795…`, both created 2026-10-15 before this run.
- `lms ps` before: no models loaded. Sweep check: none. Ports 1299/8311 free. `ws:retryprobe`
  absent.
- **Workspace:** `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh retryprobe`, then
  `./scripts/seed_demo.sh retryprobe` (`rc=0`).
- **Model config:** `opencode.direct.json`, a copy of `~/.config/opencode/opencode.local.json`
  with `lmstudio.options.baseURL` = `http://localhost:1234/v1`. It is identical to Half A's
  direct copy. `models.json` is a byte copy of `config/models.json` (`cmp` clean).
- **Server:** uvicorn **PID 144844**, a child of wrapper shell 144842, on `127.0.0.1:8311`, with
  `--reload` off and `--log-config` → a root-at-INFO file dictConfig (F-1). Same env as test plan
  §4.
  - `server.log` shows `Application startup complete` at **2026-10-16 08:46:12.067**.
    `ps -o lstart` reads 08:47:04 for 144844 and 08:46:11 for the wrapper; WSL2's `lstart` is
    skewed.
  - `retry.py` mtime is **2026-10-15 14:40:22** (epoch 1792086022). Every start reading is after
    it, so the process ran the K-067 code. The shared `:8200` server (PID 1882) was never touched.
- **LM Studio:** `lms log stream --source server --json` ran from 08:46:38 to teardown, so it had
  been running for ≥ 60 s before every leg. A second stream, `--source model --json`, ran
  alongside. It logged 19 `llm.prediction.input` and 19 `llm.prediction.output` lines
  (08:54:08–09:07:44), none during TP-101's aborted prediction. The leg trigger did not depend on it; it used `lms ps`
  status plus the server stream.
- **Human fallback:** never needed. Every `lms.exe` call returned within 0.7 s; the read-only
  `ps` hiccup is covered in B.3, TP-102.

### B.2 Results

| ID | Leg | Via | Pre-leg checks | First-attempt `kind`/`status`/`detail` prefix | Attempts / recovery | Final status | `ws:agent-team` Δfailed | Result |
|---|---|---|---|---|---|---|---|---|
| TP-101 | (e) unload mid-generation | REST | 3/3 pass (08:49:36; control seen, 60 s quiet) | extract: `kind=http status=400`, detail `{"error":"Model unloaded by user or API request."}` (classified permanent: no `falkorchat.retry` line) | 1 attempt; **no retry**, so no JIT-reload recovery | `failed`/0 at 7.56 s; embed done, 0 `ABOUT` | 0 (14→14) | **fail** (D-2) |
| TP-102 | (f) server bounce 5 s | REST | 3/3 pass (08:54:42) | extract attempt 1: `kind=network status=None`, `RemoteDisconnected: Remote end closed connection without response`; attempt 2: `kind=connection`, `connection failed: [Errno 111] Connection refused` | 3 attempts (WARNING 1/4 `retrying in 2.8s`, 2/4 `retrying in 10.6s`); attempt 3 succeeded after the restart | `ready`/0 at 33.83 s; 25 `ABOUT` | 0 (14→14) | **pass** |
| TP-103 | (f) server bounce 12 s | REST | 3/3 pass (08:57:06) | extract attempt 1: `kind=network`, `RemoteDisconnected`; attempt 2: `kind=connection`, `[Errno 111] Connection refused` | 3 attempts (1/4 `retrying in 2.6s`, 2/4 `retrying in 10.1s`); attempt 3 reached LM Studio ≈0.33 s after `Server started.` and succeeded | `ready`/0 at 32.86 s; 25 `ABOUT` | 0 (14→14) | **pass** (intent "lands between attempts 3 and 4" not realized; see B.3) |
| TP-104 | (g) contention | MCP | 3/3 pass (09:01:21) | 7 of 8 extract jobs: `kind=http status=500`, detail `<!DOCTYPE html>…<pre>Internal Server Error</pre>`; all 8 embed jobs and 1 extract job: no failure | 7 extract jobs ×2 attempts (attempt 2 succeeded); 9 jobs ×1 | `ready`/0 at **53.6 s** (wall time); every chunk embedded, 16–20 `ABOUT` each | 0 (14→14) | **pass** — "converged in this run" (n = 1) |
| TP-105 | (g) contention | REST | 3/3 pass (09:04:36) | none: all 16 jobs succeeded on attempt 1 (no `falkorchat.retry` line, no traceback) | 16 jobs ×1 | `ready`/0 at **123.51 s** (wall time); every chunk embedded; 7 chunks 16–17 `ABOUT`, chunk 1 0 (model output, see B.3) | 0 (14→14) | **pass** — "converged in this run" (n = 1); classifier not exercised |

### B.3 Per-leg evidence

Scratch evidence for every Half B leg is under
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/u5b/`:
`driver_b.py`, `precheck.py`, `commands.log`, `runs/TP-1NN.{json,out}` and
`runs/TP-1NN-{pre,post}check.json`, `server.log`, `lms-server-stream.jsonl` (`lms log stream
--source server --json`) and `lms-model-stream.jsonl` (`--source model`).

**What a request looks like in `lms log stream --source server --json`** (P3-2's open point,
answered): the positive-control `GET /v1/models` logs one `{"type":"server.log"}` line,
`[INFO] Returning {"data": [...`. A `POST /v1/embeddings` logs `Received request to embed: <text
prefix>` then `Returning embeddings (not shown in logs)`. A `POST /v1/chat/completions` logs
`[INFO][qwen/qwen3-4b-2507] Running chat completion on conversation with 2 messages.` then
`Prompt processing progress: 0.0%` / `100.0%`. Every `lms.exe` call also logs
`[LMSAuthenticator][Client=lms-cli]…` lines in the same stream, so QA's own `lms ps` polling during a
leg is visible there; it never ran inside a pre-leg quiet window.

**TP-101 (e) unload mid-generation, REST.** Nonce `caf6d57a`, document
`b1612165d50e4444b726077b81712399`, chunk `8d96db59c81f4f59976a93568056637e` (898 chars).
- Times are seconds after the ingest call. Ingest response at 0.035 s. The embed job ran first and
  finished at 2.54 s (`pendingJobs` 2→1). LM Studio logged `Running chat completion` at ≈5.2 s and
  `Prompt processing progress: 100.0%` at ≈5.4 s. `lms ps --json` showed the chat model with
  `"status": "generating"` at 5.37 s (the trigger), so the unload was mid-generation.
- `lms.exe unload qwen/qwen3-4b-2507` started at **6.869 s** (wall 08:50:52.821), returned
  `rc=0` in 0.663 s: `Model "qwen/qwen3-4b-2507" unloaded.`
- LM Studio side (server stream): `[LMSAuthenticator][Client=lms-cli][Endpoint=unloadModel]
  Unloading model: qwen/qwen3-4b-2507`, then
  `[ERROR][qwen/qwen3-4b-2507] Error: Channel Error at _0x5d72b5.<computed> (…index.js:1140:10772)`.
- falkor-chat side: the **only** line for this chunk is the job's traceback. No `falkorchat.retry`
  line, so the error was classified non-transient on attempt 1 (§3.6: a non-transient error
  re-raises with no log line):

```
2026-10-16T08:50:53.177 ERROR falkorchat.background [AnyIO worker thread] background extract failed (chunkId=8d96db59c81f4f59976a93568056637e)
    | urllib.error.HTTPError: HTTP Error 400: Bad Request
    | falkorchat.transport.ProviderCallError: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1234/v1/chat/completions: HTTP 400 Bad Request: {"error":"Model unloaded by user or API request."}
```

- State: `processing`/2 → `processing`/1 at 2.54 s → `failed`/0 at 7.56 s. The chunk has its
  embedding and 0 `ABOUT` edges. Orphans 0→0. REST `get_document` returns `status: failed` with the
  text intact.
- `lms ps` after the leg: `[]`. The chat model was not JIT-reloaded, because nothing retried.
- Scored **fail**. The leg exists to show whether the classifier recovers from LM Studio's real
  unload shape, and it does not (D-2).

**TP-102 (f) server bounce 5 s, REST.** Nonce `52e78ff1`, document
`84050bc9140a4c2db9aeafc7072f26da`, chunk `a795c58ba6af43dd9db9708f66edbc36`.
- **Void run first (not scored).** Nonce `c33d873d`, document `23de3916…`. One `lms ps --json`
  poll returned `rc=1 Error: Model does not exist.` while the chat model was being JIT-loaded. The
  driver treated that as the human-fallback condition and never bounced the server. The document
  went `ready` with no fault applied. **Fix (harness only):** a failed *read-only* `ps` poll is
  now logged and re-polled. The §6 human fallback is reserved for the fault action itself
  (unload / stop / start) erroring, or any call hanging > 20 s. Re-run with a fresh nonce after
  fresh pre-leg checks.
- Times are seconds after the ingest call; the LM Studio stream's clock runs about 0.5 s ahead of
  WSL's. Embed done by 0.53 s. LM Studio logged `Running chat completion` at ≈2.65 s and `Prompt
  processing progress: 100.0%` at ≈2.82 s. `lms ps` showed `"status": "processingPrompt"` at the
  2.59 s trigger.
- `lms.exe server stop` started at **4.094 s**, `rc=0` in 0.342 s: `Stopped the server on port
  1234.` LM Studio side: `[INFO] Server stopped.`, then `[LM STUDIO SERVER] Client disconnected.
  Stopping generation... (If the model is busy processing the prompt, it will finish first.)`.
- After a 5 s wait, `lms.exe server start` started at **9.436 s**, `rc=0` in 0.386 s: `Success!
  Server is now running on port 1234`. LM Studio logged `Server started.` /
  `Just-in-time model loading active.` at ≈10.2 s.
- falkor-chat side, the full retry trail for this chunk:

```
08:55:50.592 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=a795c58b…, kind=network, status=None), attempt 1/4, retrying in 2.8s: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1234/v1/chat/completions: RemoteDisconnected: Remote end closed connection without response
08:55:53.357 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=a795c58b…, kind=connection, status=None), attempt 2/4, retrying in 10.6s: … connection failed: [Errno 111] Connection refused
```

- Attempt 3 arrived at LM Studio at ≈18.4 s (`Running chat completion`), after the server was back.
  It completed at ≈33.5 s (`Generated prediction`). There was no further falkor-chat line, so
  attempt 3 succeeded. Recovery came at attempt 3, inside the 2/10/40 s schedule.
- State: `processing`/2 → /1 at 0.53 s → `ready`/0 at 33.83 s. 25 `ABOUT` edges, embedding
  present, orphans 0→0. REST `search_documents` returns this chunk.
- The chat model stayed loaded across the server stop (`lms ps` after shows it). Both
  `lms log stream` processes survived the stop.

**TP-103 (f) server bounce 12 s, REST.** Nonce `fe304da3`, document
`69e83f08e7b645e3816dcfce5e613bf9`, chunk `0de18a131f5e4e6c8eee904f65c48915`.
- `lms ps` showed `"status": "generating"` at the 2.577 s trigger. `server stop` started at
  **5.313 s** (`rc=0`, 0.333 s). `server start` started at **17.646 s** (`rc=0`, 0.321 s).
- LM Studio side: `Server stopped.` / `Client disconnected. Stopping generation...` at ≈4.94 s
  (stream clock), then `Server started.` at ≈17.74 s.
- falkor-chat side:

```
08:58:14.795 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=0de18a13…, kind=network, status=None), attempt 1/4, retrying in 2.6s: … RemoteDisconnected: Remote end closed connection without response
08:58:17.367 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=0de18a13…, kind=connection, status=None), attempt 2/4, retrying in 10.1s: … connection failed: [Errno 111] Connection refused
```

- Attempt 3 logged `Running chat completion` at ≈18.07 s, **0.33 s after** `Server started.`
  (≈17.74 s), and succeeded (`Generated prediction` at ≈32.7 s). State: `ready`/0 at 32.86 s,
  25 `ABOUT`, orphans 0→0. REST search returns the chunk.
- **The plan's "lands between attempts 3 and 4" did not happen.** The 12 s outage ended 0.33 s
  before attempt 3, so attempt 4's 40–60 s recovery path was **not exercised live**. Attempt 3
  fires at cumulative 10–15 s after attempt 1 fails (§3.2 jitter), and the outage ended at about
  attempt 1 + 12.9 s, so whether a 12 s wait lands before or after attempt 3 is close to a coin
  flip. Scored **pass** against its stated expectation ("as TP-102"). The plan says "repeat once",
  so it was not repeated; a deterministic attempt-4 leg needs a wait ≥ 16 s (recommendation
  B-R2 in B.6).

**TP-104 (g) unproxied contention, MCP.** Nonce `51ebc89c`, document
`9f00ed95c5464d5f933e98a3e6b3d5b6`, 8 chunks, 16 jobs.
- `lms ps` before the leg showed only `text-embedding-qwen3-embedding-0.6b` (idle). TP-103's
  consumer-half search JIT-loaded it at 08:58:46, after TP-103's `ps_after` read. `lms.exe unload
  --all` ran 0.33 s before the ingest: `rc=0`, `Unloading "text-embedding-qwen3-embedding-0.6b"...
  Unloaded 1 model.` Both models therefore started cold.
- MCP ingest response at 0.40 s, `processing`/16. All 8 embed jobs finished by 1.9 s
  (`pendingJobs` 8). The extract jobs finished one by one from 24.77 s to 53.6 s (`ready`/0).
- **The only failures in the run:** seven extract jobs, on seven distinct MCP daemon threads
  (`Thread-4/6/8/10/12/14/16 (_safe_extract)`), got `HTTP 500` within 25 ms of each other
  (09:02:25.861–.886), while the cold chat model was being JIT-loaded. The body is an Express
  default error page. LM Studio's server log carries **no line** for these seven requests; it
  shows only the eight embed requests at that moment, then eight `Running chat completion` lines
  at 2.6–3.5 s (stream clock):

```
09:02:25.861 WARNING falkorchat.retry [Thread-4 (_safe_extract)] transient provider error (extract chunkId=f9232f99…, kind=http, status=500), attempt 1/4, retrying in 2.3s: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1234/v1/chat/completions: HTTP 500 Internal Server Error: <!DOCTYPE html>
<html lang="en"> <head> <meta charset="utf-8"> <title>Error</title> </head> <body> <pre>Internal Server Error</pre> </body> </html>
```

- Jitter spread the seven retries over 2.1–3.0 s, and each succeeded on attempt 2. No give-up, no
  gate INFO, no timeout, no traceback.

| chunk | job | first-attempt failure (`kind`/`status`: detail prefix) | attempts | outcome |
|---|---|---|---|---|
| `8ab5ebde…` | extract | — | 1 | succeeded |
| `8ab5ebde…` | embed | — | 1 | succeeded |
| `f9232f99…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `f9232f99…` | embed | — | 1 | succeeded |
| `2bd49a84…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `2bd49a84…` | embed | — | 1 | succeeded |
| `80407eaa…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `80407eaa…` | embed | — | 1 | succeeded |
| `297b4b17…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `297b4b17…` | embed | — | 1 | succeeded |
| `fba9df18…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `fba9df18…` | embed | — | 1 | succeeded |
| `17527355…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `17527355…` | embed | — | 1 | succeeded |
| `eb42b26a…` | extract | http/500: HTTP 500 Internal Server Error: <!DOCTYPE html> | 2 | succeeded |
| `eb42b26a…` | embed | — | 1 | succeeded |

**TP-105 (g) unproxied contention, REST.** Nonce `aefd493c`, document
`7421eb7ae67d42d6af269e3f3167e628`, 8 chunks, 16 jobs.
- `lms ps` before: both models loaded (TP-104's). `lms.exe unload --all` ran 0.33 s before the
  ingest: `rc=0`, `Unloading "qwen/qwen3-4b-2507"... Unloading
  "text-embedding-qwen3-embedding-0.6b"... Unloaded 2 models.`
- REST response at 0.027 s, `processing`/16. The sequential `BackgroundTasks` chain then alternated
  embed and extract per chunk: `pendingJobs` stepped 16→15 at 1.53 s, →14 at 12.56 s, →13 at
  14.06 s, …, →1 at 107.5 s, and reached `ready`/0 at **123.51 s**.
- **No failure of any kind.** The server log has no `falkorchat.retry` line and no `background …
  failed` line for any of the 8 chunks. All 16 jobs succeeded on attempt 1, so this run does not
  exercise the classifier.
- **Chunk 1 (`953f8c2c…`) has 0 `ABOUT` edges**, though its extract job succeeded. LM Studio's
  `Generated prediction` for it (09:05:55) carries malformed JSON: one entity reads `"current
  speed", "type": "Concept"}`, missing its `{"name":` opener. Extraction degrades an unparseable
  reply to empty by design (plan §3.1, "model-output problems are not exceptions"). This is not a
  K-067 finding. The other 7 chunks have 16–17 `ABOUT` each, and REST `search_documents` returns
  all 8 chunks.
- `lms ps` after: only `qwen/qwen3-4b-2507` (idle); the embedding model was no longer listed.

| chunk | job | first-attempt failure | attempts | outcome |
|---|---|---|---|---|
| `953f8c2c…` | extract | — | 1 | succeeded |
| `953f8c2c…` | embed | — | 1 | succeeded |
| `bc28910b…` | extract | — | 1 | succeeded |
| `bc28910b…` | embed | — | 1 | succeeded |
| `a601f93c…` | extract | — | 1 | succeeded |
| `a601f93c…` | embed | — | 1 | succeeded |
| `cd4d290b…` | extract | — | 1 | succeeded |
| `cd4d290b…` | embed | — | 1 | succeeded |
| `0e9be732…` | extract | — | 1 | succeeded |
| `0e9be732…` | embed | — | 1 | succeeded |
| `12812d3f…` | extract | — | 1 | succeeded |
| `12812d3f…` | embed | — | 1 | succeeded |
| `cefddda1…` | extract | — | 1 | succeeded |
| `cefddda1…` | embed | — | 1 | succeeded |
| `47cdb35d…` | extract | — | 1 | succeeded |
| `47cdb35d…` | embed | — | 1 | succeeded |

### B.4 (g) reading, per plan §3.2.1

Each pre-stated reading (test plan §6), applied to TP-104 and TP-105:

| Pre-stated reading | Observed | Applies? |
|---|---|---|
| Any embed job with `kind=timeout` → apply §3.2.1 (flip `EMBED_CHUNK_RETRY.retry_timeouts`, re-run) | 0 of 16 embed jobs failed at all; no `kind=timeout` line and no `TimeoutError` / `timed out` traceback anywhere | **No.** No flip, no re-run. |
| A transient-looking 4xx body matching no marker → add the marker, re-run | (g)'s only failures were `HTTP 500` (classified transient by the 5xx rule); no 4xx in (g) | **No** (but see D-2: such a body *was* captured in (e)) |
| A raw non-`ProviderCallError` in the log → return to `architect` | none: no `background … failed` line at all in (g) | **No** |
| `ready` → "converged in this run" (n = 1), never a convergence claim | TP-104 `ready` in 53.6 s; TP-105 `ready` in 123.51 s | **Yes**, for both |
| `failed` with transient kinds → mechanism delivered; F-4 filed before K-067 leaves `BACKLOG.md` | neither run ended `failed` | No |

- **§3.2.1 condition 3 (P3-1):** not applicable, because there is no (g) re-run. For the record,
  the independent set (embed tracebacks whose `ProviderCallError` contains `TimeoutError` or
  `timed out`, plus embed lines carrying `kind=timeout`) is **empty** in both TP-104 and TP-105.
- **K-067 closure rule:** the final (g) outcome is `ready`, so the report says **"converged in
  this run"** and the F-4-before-closure branch does not fire. That reading covers only (g).
  D-2 is a separate open question about classifier fit.
- **What the classifier fit on in (g):** LM Studio's cold-load fault under 8 concurrent chat
  requests is an `HTTP 500` whose body is an Express default error page
  (`<title>Error</title> … <pre>Internal Server Error</pre>`). LM Studio's own server log has no
  line for it. The status rule caught it, so a body marker could never have been needed for it.

### B.5 Teardown

- uvicorn 144844 stopped: `Finished server process [144844]` (09:09:46). Both `lms log stream`
  processes stopped. Ports 1299 and 8311 are free. `:8200` (PID 1882) was untouched and is still
  listening.
- `redis-cli GRAPH.DELETE ws:retryprobe` → `OK`, and `GRAPH.LIST` no longer shows it. Before the
  delete it held 5 `ready` and 1 `failed` (TP-101), with orphan count 0.
- **`lms ps` restored.** It was `[]` before the run. At teardown it showed only
  `text-embedding-qwen3-embedding-0.6b` (TP-105's consumer search had JIT-loaded it). `lms.exe
  unload --all` → `Unloaded 1 model.`, then `lms ps --json` → `[]`. The LM Studio server is still
  running on 1234: it was stopped/started twice in (f) and left running.
- **`ws:agent-team` re-checks:** after every leg and at teardown (09:10:05): `processing` 0,
  `failed` 14, `ready` 340. **The failed delta was 0 at every check**, with no new failed
  documents (baseline list `agentteam_failed_baseline.txt`).
- The shared `~/.config/opencode/opencode.json` and the shipped `config/models.json` are
  untouched. No commit was made and no tree-mutating git command was run.
- No permission denials occurred.
- **Post-teardown side effect, caused by QA's learning capture.** After teardown, QA filed two
  raw-capture documents into `ws:agent-team` via `ingest_document` on the shared `:8200` server.
  That server runs pre-K-067 code.
  - `69d50ac7f38043e48e39edbef7e6293d` (the `lms ps` / log-stream fact) → `ready`.
  - **`9b2c6b30c7654ed49655210b45a29d59`** (title begins *"LM Studio answers an in-flight chat
    completion with HTTP 400 …"*) → **`failed`**. The `:8200` server configures no logging
    (F-1), so its cause is not visible from here.
  - `ws:agent-team` at 09:12:45 reads `processing` 0, **`failed` 15** (+1 vs the 14 baseline,
    exactly this document), `ready` 341.
  - Per §6, the coordinator owns its re-drive; QA did not re-drive it.
  - These ingests JIT-loaded both models. A second `lms.exe unload --all` restored `lms ps` to
    `[]`.

### B.6 Defects, coverage, and recommendations

**D-2 — LM Studio's real mid-request unload response is classified permanent, so the document
fails on attempt 1 with no retry** (severity: **medium**; K-067 classifier-fit gap, review M-3 /
risk R7).
- **Steps:**
  1. Ingest a 1-chunk document via REST `POST /documents`.
  2. When `lms ps --json` shows `qwen/qwen3-4b-2507` with `"status": "generating"`, run
     `lms.exe unload qwen/qwen3-4b-2507`.
- **Expected** (plan §5 (e), "the closest reproducible relative of `terminated`"): the in-flight
  failure is retried, and attempt 2 recovers after the JIT reload.
- **Actual:**
  - `ProviderCallError … HTTP 400 Bad Request: {"error":"Model unloaded by user or API request."}`;
  - no `falkorchat.retry` line, because `is_transient` returned False (4xx with no marker);
  - `failed`/0 at 7.56 s (TP-101, B.3).
- **LM Studio side:** it logs `[ERROR][qwen/qwen3-4b-2507] Error: Channel Error` for this. The
  plan's inferred `channel error` marker (§3.1) is therefore real LM Studio text, but in this
  observation it stays in LM Studio's own log and never reaches the HTTP body.
- **Impact:** any unload of the chat model while an extract job is in flight fails the whole
  document, even though the model JIT-reloads in about 2–3 s. That covers a human Eject and
  `lms unload`. Whether LM Studio's own TTL or JIT auto-evict produces the same body was not
  tested.
- **The plan fork (coordinator's call; QA did not edit source):**
  - Plan §3.2.1 pre-authorizes adding a marker for a 4xx body "whose body names a load, crash, or
    serving failure", but it scopes that to bodies **captured in leg (g)**. This body came from
    (e).
  - **If adopted, the prescribed change is:** add `"model unloaded"` to
    `retry.TRANSIENT_DETAIL_MARKERS`, with a U-5 row (R-1), then re-run TP-101 with a fresh nonce
    on a server started after the new `retry.py` mtime.
  - **The trade-off to weigh:** the body says *"by user or API request"*. Retrying it JIT-reloads
    a model an operator deliberately ejected.

**D-1 recurred** (Half A, §4; already reported): 10 `background fuse failed` / `RediSearch:
Syntax error` lines in this run's `server.log`, including a new `near Vell` variant. There is no
new defect.

**Coverage:**
- **Live now:**
  - classifier fit for server stop: `RemoteDisconnected` → `network`, refused → `connection`
    (TP-102, TP-103);
  - the cold JIT-load `500` under MCP concurrency (TP-104);
  - classifier fit for mid-request unload, as a gap (TP-101);
  - the REST and MCP (g) paths end to end.
- **Not covered:**
  - attempt-4 recovery on a real fault (TP-103's restart landed 0.33 s before attempt 3);
  - any embed `kind=timeout`, never observed, so the §3.2.1 flip stays untriggered;
  - a classifier-exercising REST (g): TP-105 had no faults;
  - (e) over MCP;
  - repeatability: every (g) result is n = 1.

**Acceptance mapping, Half B row** ("classifier fires on real LM Studio faults"): **partially
evidenced.** It fires on a server stop and on the cold-load `500`. It does **not** fire on a
mid-request unload (D-2). This row stays open until D-2 is decided.

**Recommendations:**
- **B-R1:** decide D-2 per the fork above (`teco`, with `architect` if the "by user" trade-off
  needs design judgment). Then re-run TP-101.
- **B-R2:** a deterministic live attempt-4 leg needs an outage that ends after attempt 3's latest
  firing point, which is 15 s after attempt 1 fails. So use a wait of about 16–35 s, not 12 s.
- **B-R3 (harness):** `lms ps --json` can return `rc=1 Error: Model does not exist.` during a JIT
  load. Any Half B harness must treat a failed read-only poll as poll noise, not as the §6
  human-fallback trigger.
- **B-R4:** in `lms log stream --source server`, every `lms.exe` call logs
  `[LMSAuthenticator][Client=lms-cli]` lines. A quiet-window check must not run concurrently
  with `lms` polling, and `precheck.py` never does.

### B.7 TP-101 re-run after D-2 fix (21ff9f9b)

**Result: pass. D-2 is closed.** The fix adds `"model unloaded"` to
`retry.TRANSIENT_DETAIL_MARKERS` (`server/falkorchat/retry.py:55`).

- **Wiring:**
  - HEAD `21ff9f9b`; `retry.py` mtime **2026-10-16 09:17:46** (epoch 1792153066).
  - Fresh uvicorn **PID 150688** on `127.0.0.1:8311`, with the same env, configs and
    `--log-config` as B.1. Its `ps -o lstart` reads **09:18:18**, and `server.log` shows
    `Application startup complete` at 09:18:18.452. Both are after the mtime.
  - `ws:retryprobe` was re-created with `bootstrap_schema.sh` and `seed_demo.sh` (`rc=0`).
  - Both `lms log stream` processes were restarted at 09:18:24.
  - The run-1 `server.log` was kept as `server.halfB-run1.log`. QA's mistake: the stream restart
    overwrote run 1's raw `lms-*-stream.jsonl` files. Each run-1 leg's stream lines remain in its
    `runs/TP-10N.json` (`lms_server_stream`), which is what B.3 quotes.
- **Coordinator N-7:** `teco` confirmed that no other coordinated agent was in flight.
- **Pre-leg checks (09:18:38), 3 of 3 pass:**
  - `ws:agent-team` `processing` 0, `failed` **15**. The new baseline is the old 14 plus QA's own
    `9b2c6b30…` capture (B.5); `comm` against the 14-list shows exactly that one ID added.
  - The sweep check `pgrep` returned nothing.
  - The positive-control line was seen, followed by 60 s of quiet.
- **Leg:** nonce `d659a408`, document `4bfb7f6457a94247ab7e76ffadcc2ef3`, chunk
  `bb015cc9ada34c8bba60be4d020b69d5`. Times below are seconds after ingest.
  - The embed job finished by 1.05 s.
  - LM Studio logged `Running chat completion` at ≈3.39 s and `Prompt processing progress: 100.0%`
    at ≈3.56 s.
  - `lms ps --json` showed `"status": "generating"` at the 3.79 s trigger.
  - `lms.exe unload qwen/qwen3-4b-2507` started at **5.287 s**: `rc=0` in 0.607 s, `Model
    "qwen/qwen3-4b-2507" unloaded.`
  - LM Studio logged `Unloading model` and then `[ERROR][qwen/qwen3-4b-2507] Error: Channel Error`,
    the same as in run 1.
- **falkor-chat side.** Now classified transient:

```
2026-10-16T09:19:48.744 WARNING falkorchat.retry [AnyIO worker thread] transient provider error (extract chunkId=bb015cc9ada34c8bba60be4d020b69d5, kind=http, status=400), attempt 1/4, retrying in 2.6s: lmstudio/qwen/qwen3-4b-2507 @ http://localhost:1234/v1/chat/completions: HTTP 400 Bad Request: {"error":"Model unloaded by user or API request."}
```

- **JIT reload.** About 2.6 s after the WARNING, attempt 2 reached LM Studio as `[qwen/qwen3-4b-2507]
  Running chat completion` (≈10.40 s, stream clock), on the model that had just been unloaded. It
  completed with `Generated prediction` at ≈25.48 s. `lms ps` after the leg shows
  `qwen/qwen3-4b-2507` loaded (`idle`). The server log has no further line for this chunk: no
  traceback, no give-up.
- **State:** `processing`/2 → /1 at 1.05 s → **`ready`/0 at 26.48 s**. 26 `ABOUT` edges, embedding
  present, orphans 0→0. REST `search_documents` returns the chunk.
- **The three done-conditions:** an `attempt 1/4` WARNING, a JIT reload, and `ready`. All three are
  observed.
- **Post-leg `ws:agent-team` check (09:20:20):** `processing` 0, `failed` 15, `ready` 341. The
  failed delta was **0**.
- **Acceptance-mapping row** ("classifier fires on real LM Studio faults"): now **evidenced** for
  the three shapes Half B observed:
  - mid-request unload `400` (this re-run);
  - server-stop `network` and `connection` (TP-102/103);
  - cold-load `500` (TP-104).

  Still not covered: any embed `kind=timeout`, and attempt-4 recovery on a real fault (B.6).
- **Teardown (09:29):**
  - uvicorn 150688 stopped (`Finished server process [150688]`), and both stream processes
    stopped. Ports 1299 and 8311 are free. `:8200` (PID 1882) was untouched and is still
    listening.
  - `GRAPH.DELETE ws:retryprobe` → `OK` (it held 1 `ready` document), and it is gone from
    `GRAPH.LIST`.
  - `lms.exe unload --all` → `Unloaded 1 model.` (the search had JIT-loaded the embedding model).
    `lms ps --json` → `[]`, the same as before the re-run. The LM Studio server is running on
    1234.
  - Final `ws:agent-team` check (09:29:10): `processing` 0, `failed` 15, `ready` 341. The delta
    was 0.
  - No `ingest_document` capture was filed, and no commit or tree-mutating git command was run.

