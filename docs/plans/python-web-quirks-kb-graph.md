# Python-web-quirks KB: test workspace verification (S6a, S6b)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** python-web-quirks-kb (S6a, S6b)

Change against `docs/plans/python-web-quirks-kb.md` §D11 "Test workspace". Observed 2026-11-10 on `falkordb/falkordb:v4.18.11`. Every figure below is from command output in this run.

## 0. Starting state
- `redis-cli GRAPH.LIST`: `cpg_falkorchat ws:nlq-eval ws:eval ws:acme ws:demo ws:agent-team ws:k069probe reference`. No `ws:pwq-test`.
- `:8200/health` -> `{"status":"ok"}` (pid 1268); `:8210/health` -> no answer.
- `ws:agent-team` (read only): 482 Documents (all `currentVersion`), 773 Chunks, 757 with `embedding`; 16 unembedded = 1 chunk of a `processing` document + 15 chunks of 14 `failed` documents.

## 1. S6a verdict: `GRAPH.COPY` carries the vector index and the embeddings

Probe keys `pwqprobe_src`, `pwqprobe_dst` (both deleted at the end, `GRAPH.DELETE` returned OK for each).

```
GRAPH.QUERY pwqprobe_src "CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding) OPTIONS {dimension:4, similarityFunction:'cosine'}"
GRAPH.QUERY pwqprobe_src "CREATE (:Chunk {id:'a', embedding: vecf32([1.0,0.0,0.0,0.0])}), (:Chunk {id:'b', embedding: vecf32([0.0,1.0,0.0,0.0])}), (:Chunk {id:'c'})"
GRAPH.COPY pwqprobe_src pwqprobe_dst
```

Results on the copy:
- `CALL db.indexes()` -> `Chunk`, `[embedding]`, `{embedding: [VECTOR]}`, options `{dimension: 4, similarityFunction: cosine, M: 16, efConstruction: 200, efRuntime: 10}`, status `OPERATIONAL`, `numDocuments: 2`.
- `CALL db.idx.vector.queryNodes('Chunk','embedding',3,vecf32([1.0,0.1,0.0,0.0]))` -> `a 0.00496`, `b 0.90050`.
- `MATCH (c:Chunk) RETURN count(c), count(c.embedding)` -> `3, 2` (source identical).
- `GRAPH.COPY` onto an existing destination -> `destination key already exists`.

**Verdict: yes.** Index definition and embeddings both survive; D11's `GRAPH.COPY` path applies, no re-ingest needed.

Two operational findings:
1. The first `GRAPH.COPY` attempt returned `GRAPH.COPY failed, could not fork` (host `free -m`: 121 MB free, `overcommit_memory` 0; `latest_fork_usec` 4143 afterwards). The retry seconds later returned OK. A copy can therefore fail transiently; retry once and check `GRAPH.LIST` before retrying.
2. I then queried `pwqprobe_dst` before retrying, which silently created it (a write against a missing key creates the graph) and the retry would have hit `destination key already exists`. I deleted that stray probe key and re-ran. For `ws:pwq-test` I did not query the destination between attempts.

## 2. S6b: `ws:pwq-test` creation and server start

Order per D11: `GRAPH.LIST | grep -c pwq-test` -> `0`, then `GRAPH.COPY ws:agent-team ws:pwq-test` -> `OK` (first attempt), then the server.

On `ws:pwq-test` right after the copy (before the server started):
- Documents: 482 current; Chunks: `773, 757` (identical to production). **Unembedded chunks: 16** = `processing` 1 chunk / 1 document, `failed` 15 chunks / 14 documents.
- `CALL db.indexes()` filtered to `Chunk` -> `{chunkId: [RANGE], embedding: [VECTOR]}`, `OPERATIONAL`.

Server (additive; `:8210` was not up, so nothing restarted):
```
cd falkor-chat; mkdir -p ~/.local/state/graphmind-ai-lab
FALKORCHAT_WS_ID=pwq-test AGENT_TEAM_PORT=8210 setsid nohup ./scripts/start_agent_team.sh \
  > ~/.local/state/graphmind-ai-lab/pwq-test-8210.log 2>&1 < /dev/null &
```
- Log: `[3/5] Bootstrapping schema for workspace 'pwq-test'`, indexes reported `already indexed`, `ws:pwq-test already pinned to 'lmstudio/text-embedding-qwen3-embedding-0.6b' — no-op`, `Uvicorn running on http://127.0.0.1:8210`. Bootstrap and seed were idempotent on the copied graph.
- `ss -ltnp`: `:8200` pid 1268 (untouched), `:8210` pid 1815524 (the new uvicorn; the launching shell job exited, the process survived).
- Post-start the graph is unchanged: `773 / 757` chunks, `Chunk` index still `[RANGE, VECTOR]`; `WorkspaceConfig` `default` on both graphs.
- Log path for `teco`'s liveness check: `~/.local/state/graphmind-ai-lab/pwq-test-8210.log`.

Known-answer search, query built with the `agent-kb-retrieval` prefix, `GET /documents/search?limit=5`, query "FalkorDB vector index must be created before the uniqueness constraint": `:8210` and `:8200` returned the identical five chunk lists, top `d624e246... 0.3271`, then `0.371`, `0.3758`, `0.3899`, `0.4007`. (The REST search hit carries no `title`, prints `None` in my helper.)

## 3. F6: URL-to-workspace proof

Test document POSTed to `:8210` only (title `PWQ-PROBE alpha`, text with `⊂`, a unique phrase `zebra-quartz-meridian`, trailing newline, `producedBy: cobb`) through `pwq_kb.RestClient(check_target("pwq-test","http://localhost:8210"))`:
- Receipt: `{"documentId": "744975e7f33849a2a1a8a678837d77dc", "chunkCount": 1, "status": "processing", "autoSuperseded": false, "supersededDocumentId": null}`. HTTP 201 (the client accepts only 201).
- `CypherFetcher.current_docs("pwq-test", [title], [id])` -> the document, status `processing` then `ready`. `current_docs("agent-team", [title], [id])` -> `[]`.
- Graph reads: `ws:pwq-test` count of documents matching id/title/unique phrase `1`, `ws:agent-team` `0`.
- `GET :8210/documents/search` for the unique phrase: top hit `744975e7... 0.4482`; `:8200` same query: top hit `7f0d6442... 0.5384`, the probe document absent. 

**Verdict: `:8210` writes to `ws:pwq-test` and never to `ws:agent-team`.** The mapping holds for this launch (env `FALKORCHAT_WS_ID=pwq-test` echoed in the startup banner `Workspace: ws:pwq-test`). The tool still never verifies it; the plan's R-after-first-POST fetcher check (F6 suggestion) is the runtime guard.

## 4. Target guard (`claude/cobb/scripts/pwq_kb.py`, `check_target`, `TARGETS`)

Calls made in this run:
- `check_target("pwq-test","http://localhost:8210")` -> `Target(ws='pwq-test', base_url='http://localhost:8210', production=False)` (accepted; trailing slash `.../8210/` also accepted).
- `check_target("agent-team","http://localhost:8200")` -> `TargetRefused: production ingest needs --confirm-production`; with `confirm_production=True` -> `Target(..., production=True)`.
- `check_target("pwq-test","http://localhost:8200")` -> `TargetRefused: --base-url 'http://localhost:8200' does not match the target table entry for --ws pwq-test (http://localhost:8210)`.

These were pure function calls; no request was made to `:8200`.

## 5. Review S6a list, on `:8210` only

1. **POST of one unit, `producedBy=cobb`:** HTTP 201, receipt above. Provenance is edge `(Document)-[:INGESTED_BY]->(Agent {agentId:'cobb'})`; `cobb` already existed in `ws:pwq-test` via the copy (roster: teco, tico, architect, coder, tdd-engineer, frontend-engineer, qa-engineer, analyst, data-scientist, graph-dba, devops, security-expert, cobb). The `Document.producedBy` property itself is null.
2. **Byte identity:** the posted text (`⊂`, final `\n`) compared `==` to `Document.text` read through `CypherFetcher.current_docs` -> `True`; tail `'econd line.\n'`. No normalization on store.
3. **Auto-supersede:**
   - Identical re-POST -> `{"documentId": "a29230f6...", "chunkCount": 1, "status": "processing", "autoSuperseded": true, "supersededDocumentId": "744975e7..."}`; fetcher then returned exactly one current document (`a29230f6`, `ready`); the old one has `currentVersion false`; one `SUPERSEDES` edge `a29230f6 -> 744975e7`, status `confirmed`.
   - Edited text (one title, appended words): receipt `autoSuperseded: false`; the fetcher then returned **two current documents** (`a29230f6`, `3d2d1ff0`), both `currentVersion true`, status `ready`.
   - A second probe (title `PWQ-PROBE beta`, 5 chunks, one-word edit, 10 s apart): again `autoSuperseded: false`, two current documents. After 15 s waits **no pending `SUPERSEDES` edge existed** in either probe (`MATCH ()-[r:SUPERSEDES]->() WHERE r.status<>'confirmed'` returned no rows). Observation relevant to the `differing` refusal: an edited text is a **second current document**, not a pending suggestion, in this build/corpus. Separately observed: `count(d.lshBand0)` is `0` of 487 Documents in `ws:pwq-test` (all carry `textNormalizedHash`); I did not investigate why detection created no edge.
4. **Gate during real embedding:** `run_gate` over a 14-chunk document (`gamma`), a planted failure (`delta`) and a never-existing id (`ghost`), timeout 90 s, poll 1 s: it waited while embedding (elapsed 15.9 s), then returned `gate ok: False` with `gamma 14 14 ok`, `delta 1 1 FAILED`, `ghost - - MISSING`. The failure is **planted**: I set `d.status='failed'` by a direct `GRAPH.QUERY` write on `ws:pwq-test` (the field `background._report_document_job` writes); I did not produce a genuine embedding failure. Note `gamma`'s `Document.status` was still `processing` immediately after 14/14 embedded and became `ready`, `pendingJobs 0`, about 20 s later; the gate keys on embedded counts, so it was not affected.
5. **`DELETE` response shape:** HTTP 200, `content-type: application/json`, body `{"documentId":"744975e7f33849a2a1a8a678837d77dc","deleted":true}`. Unknown id: HTTP 404 `{"error":"DocumentNotFoundError","detail":"000...0"}`. Audit: `GET /documents/{id}/deletion` -> `{"documentId":...,"deletedBy":"u1","deletedAt":1794310265931}`. The client's HTTP-200 expectation holds.
6. `replace`/`retire` round and `eval --probe/--before` (review items 5-7) were not in this task's list and were not run.

## 6. Cleanup and end state
- Deleted from `:8210` by REST `DELETE` (7 documents, titles `PWQ-PROBE alpha` x3, `beta` x2, `gamma`, `delta`; first returned the body above, the other six HTTP 200). Post-cleanup `ws:pwq-test`: 482 Documents, Chunks `773, 757`, 0 `PWQ-PROBE` documents; `ws:agent-team` has 0 `PWQ-PROBE` documents.
- `GRAPH.DELETE` was used only on `pwqprobe_src` and `pwqprobe_dst` (and the stray `pwqprobe_dst` of the failed-fork attempt). `ws:pwq-test` was not deleted; `ws:agent-team` and `:8200` were only read.
- `GRAPH.LIST` at the end: `cpg_falkorchat ws:nlq-eval ws:eval ws:acme ws:demo ws:agent-team ws:k069probe reference ws:pwq-test`. Both `/health` -> `{"status":"ok"}`.
- `:8210` stays up (pid 1815524) for S7 to S17; if it dies, restart with the command in §2 (the script is idempotent) but only after `GRAPH.LIST` confirms `ws:pwq-test` exists (never `GRAPH.COPY` over it).
- Learning-capture document written to `ws:agent-team`: `458d329e2ba44e45a5fd82301a47504a`.
