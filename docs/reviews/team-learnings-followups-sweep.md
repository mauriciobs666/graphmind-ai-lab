# Team-learnings follow-ups: sweep report

> **Status:** active · **Owner:** `cobb` · **Tracks:** K-063, K-102 (—)

Follow-ups left by `docs/reviews/team-learnings-distillation6-sweep.md` (follow-up 3), `docs/reviews/team-learnings-distillation5-sweep.md` (follow-up 7) and `docs/reviews/team-learnings-distillation4-sweep.md` (finding 3). Date: 2026-11-07. Embedder rule followed: `/api/v0/models` listed `text-embedding-qwen3-embedding-0.6b` as `loaded` before and after each ingest; one ingest at a time; each polled to `ready`.

## A. Failed raw documents: not re-embedded (stopped, decision needed)

**Observed.** `MATCH (d:Document) WHERE d.status = 'failed'` on `ws:agent-team` returns 17 documents now (the property is `Document.status`). The six named in distillation 4 finding 3 (rows 1, 12, 14, 15, 25, 40) are `4b9c6680`, `a792f59a`, `7bf10aeb`, `15de3552`, `8e05d0f8`, `2cc6debd`. They are not alike: `4b9c6680`, `15de3552` and `2cc6debd` each have a `Chunk` with no `embedding` property (invisible to `search_documents`); `a792f59a`, `7bf10aeb` and `8e05d0f8` have every chunk embedded and failed some other way (likely the extraction job; not diagnosed). Re-run on 2026-11-07: across the whole workspace 15 documents have an unembedded chunk (14 `failed`, 1 `processing`); `falkor-chat/docs/BACKLOG.md` K-102 still says 11 and was not edited (see Follow-ups).

**Re-embed path.** None that is additive and per document. Grepped `falkor-chat/server/falkorchat` and `falkor-chat/scripts`: `repository.set_chunk_embedding` and `EmbeddingWorker.embed_chunk` are internal, with no HTTP route or MCP tool calling them for an existing document, and no retry endpoint. `falkor-chat/scripts/migrate_embeddings.sh` re-embeds a whole workspace to a new model and drops and rebuilds both vector indexes; its header requires stopping the server and restarting it, so it is not additive. `ingest_document` of the same text would create a new document (the raw documents are the user's; recreating is barred). `falkor-chat/docs/BACKLOG.md` K-102 already records this exact gap (11 failed documents then, "decide whether a failed document's chunks are re-embedded") as unowned and proposed.

**Options.** (1) Leave as is: the six stay readable by `get_document` and full-text, and every one is dispositioned in the manifest. (2) Have an implementer add a documented repair operation under K-102 (re-embed unembedded chunks of a `failed` document, set status from the outcome). (3) Write the embeddings with a hand-built Cypher or a one-off script calling `set_chunk_embedding`: bypasses the dimension guard and the status bookkeeping; not done. **Recommendation:** (1) now, (2) via K-102; the three embedded-but-failed documents need a root cause, not a re-embed.

## B. KB claim `fd27e4a9`: re-ingested

Re-run: the claim's directory is the Windows-side `.lmstudio/server-logs/` (reachable from WSL under `/mnt/c/Users/<user>/`; `~/.lmstudio` does not exist in WSL). `find -type f` finds 138 files; `grep -rl 'Num Offload Layers'` finds 2: `2026-10/2026-10-30.1.log` and `2026-11/2026-11-07.1.log`. Each hit is one line of a logged request/response body holding entity-extraction output that quotes the string (read the matching lines), not a placement line. So the earlier "the one file" was a count of 1 at check time; it grows when the fact is processed again. Fixed in place in `claude/data-scientist/lm-studio-model-notes.md` (the section "LM Studio's per-load GPU placement decisions ..."). New claim `d9efef15867747fd8d5c3c22ac0d218c`: `ready`, 2 of 2 chunks embedded, `size(text)` 1581 = the `.md` body length; a full content compare was not run.

## C. `GRAPH.MEMORY USAGE` integer-MB clause

Placement read first: line 1239 is a standalone bullet, one claim, in the "Ops, config & tooling" section of `claude/graph-dba/falkordb-quirks.md`; the clause was appended to that bullet. Observed (read-only command, `falkordb-dev`, module `41811`): `total_graph_sz_mb` is 1 for `ws:acme` (874 nodes), 0 for `ws:test` (5 nodes), `ws:eval` (142) and `ws:k069probe` (18), 12 for `ws:agent-team`; every field is an integer. `ws:acme` has VECTOR indexes (`Chunk.embedding`, `Message.embedding` per `db.indexes()`) yet reads `indices_sz_mb` 0, while `ws:agent-team` reads 3; whether those indexes hold vectors was not checked. New claim `424a70a212474976bef248e17f4958f2`: `ready`, 1 of 1 chunk embedded, `size(text)` 527 = the `.md` bullet length; content compare not run.

## For the user

Delete the superseded claims (agents cannot):

```
delete_document fd27e4a9cf7b4333be7ac9f7445a0e46
delete_document 9bdc41de8e5e469687417a72314b90f6
```

Or by REST, per `falkor-chat/docs/SERVER.md` (`DELETE /documents/{id}`; the `:8200` server answered `/health`; **not run**, and any actor header it needs was not checked): `curl -X DELETE http://127.0.0.1:8200/documents/fd27e4a9cf7b4333be7ac9f7445a0e46` and the same for `9bdc41de8e5e469687417a72314b90f6`. The manifest note under `_cobb_followups_2026_11_07` names both. Decision for A: leave, or open the K-102 repair item.

## Follow-ups (not changed)

1. `falkor-chat/docs/BACKLOG.md` K-102 says 11 failed documents; the same query now returns 14 `failed` + 1 `processing`. Owner of that doc to update.
2. `claude/graph-dba/falkordb-quirks.md` states `indices_sz_mb: 0` "with a live HNSW index holding real vectors". Given `ws:acme` above, that 0 may be integer-MB rounding rather than a true under-report; graph-dba should verify against an `INFO memory` delta. Not edited here.

## Files touched

`claude/data-scientist/lm-studio-model-notes.md`, `claude/graph-dba/falkordb-quirks.md`, `claude/cobb/scripts/kb-claim-manifest.json`; kaizen: `claude/cobb/kaizen/{plan,history}.md`, `claude/graph-dba/kaizen/history.md`, `claude/data-scientist/kaizen/history.md`. No catalog changed.
