# python-web-quirks KB: 4b staging workspace (ws:pwq-test on :8210)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — (W1 of `docs/plans/python-web-quirks-kb-path-ml.md` §6)

Result and runbook of unit W1, run 2026-11-10 under the stakeholder's F3 authorisation. Scope: only `:8210` and graph `ws:pwq-test`. Nothing on `:8200` or any other workspace was written (reads only).

## Embedder and dimension evidence

`POST localhost:1234/v1/embeddings` with model `text-embedding-qwen3-embedding-4b`, input `hello`, `"dimensions": 1024` returned a vector of length **2560** (LM Studio ignores `dimensions`). Indexes are sized 2560.

## Commands run

1. Overlay (new file, production's `falkor-chat/config/models.json` untouched): `falkor-chat/config/models.pwq-staging.json` = `models.json` plus `models."lmstudio/text-embedding-qwen3-embedding-4b".dim = 2560`. Default embedder in it stays the 0.6b.
2. Stop `:8210` only (pid 1815524, found by `ss -ltnp`, a `falkorchat.app` uvicorn with `FALKORCHAT_WS_ID=pwq-test`): `kill 1815524`. `:8200` (pid 1268) kept listening.
3. Migrate (supported route):
   ```
   cd falkor-chat
   FALKORCHAT_MODEL_CONFIG=$PWD/config/models.pwq-staging.json \
   FALKORCHAT_OPENCODE_CONFIG=$HOME/.config/opencode/opencode.local.json \
   ./scripts/migrate_embeddings.sh pwq-test lmstudio/text-embedding-qwen3-embedding-4b --i-have-stopped-traffic
   ```
   Output: `Chunk: migrated 902/902 (skipped 0) in 75.0s total`, `Message: migrated 0/0`, `migrated to 'lmstudio/text-embedding-qwen3-embedding-4b' at dim 2560`. Before it, 886 of 902 chunks carried an embedding; after it, 0 chunks lack one (the 16 previously unembedded chunks were embedded too).
4. Restart `:8210` with the original environment (`FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_OPENCODE_CONFIG`, `FALKORCHAT_USER_ID=u1`, `FALKORCHAT_WS_ID=pwq-test`, `AGENT_TEAM_PORT=8210`, FalkorDB 127.0.0.1:6379) plus `FALKORCHAT_MODEL_CONFIG=<overlay above>` and `FALKORCHAT_EMBEDDING_DIM=2560`, from `falkor-chat/server`:
   `.venv/bin/uvicorn falkorchat.app:app --port 8210 --log-config ../config/logging.json` (log: session scratchpad `uvicorn8210.log`). Startup complete; new pid 3487158.
5. Load, with the work manifest `m-work.json`:
   `python3 claude/cobb/scripts/pwq_kb.py ingest --ws pwq-test --base-url http://localhost:8210 --manifest <scratchpad>/m-work.json`
   Result: all 36 units reported `skip` (already present: the migration re-embeds in place, so no document needed re-posting). Test-only documents were likewise preserved: `ws:pwq-test` holds 518 `Document` nodes and 902 `Chunk` nodes after the run.
6. `python3 claude/cobb/scripts/pwq_kb.py gate --ws pwq-test --base-url http://localhost:8210 --manifest <scratchpad>/m-work.json`: every unit row `chunks == embedded`, status `ok`, final line **`gate: clean`**, exit 0 (e.g. u01 3/3, u02 7/7, u21 9/9, u26 10/10).

## Staging workspace configuration (observed)

- `WorkspaceConfig.embeddingModelOverride` = `lmstudio/text-embedding-qwen3-embedding-4b` (set by `embedding_migration.migrate`).
- `Chunk.embedding` and `Message.embedding` vector indexes: `dimension: 2560, similarityFunction: cosine, M: 16, efConstruction: 200, efRuntime: 10`.
- All 902 chunks carry `embeddingModel` = the 4b ref.
- Production default (`config/models.json`) still names the 0.6b; `ws:agent-team` Chunk index still `dimension: 1024`, chunk count 884 before and after, override still the 0.6b.

## Rebuild

Repeat steps 2 to 4 (stop `:8210`, `migrate_embeddings.sh` is idempotent and resumable, restart with the overlay). If the workspace were wiped, bootstrap it first (`scripts/bootstrap_schema.sh`/`create_workspace.sh` at dim 2560), then run step 5 and 6. Always keep `FALKORCHAT_MODEL_CONFIG` on the overlay for this server, or the 4b dim is undeclared and embedding writes fail with `EmbeddingDimensionError`.

## State of :8210 after

Running (pid 3487158), serving `ws:pwq-test`, 4b override, 2560-dim indexes, `gate` clean. `:8200` pid 1268 unchanged. Not run: `--confirm-production`, sealed files, `eval`.
