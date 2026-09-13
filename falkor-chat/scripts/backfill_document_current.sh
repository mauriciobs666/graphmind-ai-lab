#!/usr/bin/env bash
# backfill_document_current.sh — one-off: stamp Document.currentVersion /
# Chunk.documentCurrent on content ingested before document-ingestion2 Stage A
# (commit 4a6186b, 2026-09-11).
#
# Usage:
#   ./scripts/backfill_document_current.sh <workspaceId> [<workspaceId> ...]
#
# Stage A's `create_document` is the first write path that stamps these two
# properties; every Document/Chunk ingested before it landed has NEITHER
# property set at all (absent, not `false`). Stage C's default-search filter
# (`WHERE seed.documentCurrent = true`, repository.py's `search_chunks`)
# evaluates a missing property as NULL (falsy), so `search_chunks`/
# `search_documents`/`GET /documents/search` silently return zero results
# forever for any workspace with pre-Stage-A content, until this backfill
# runs (docs/reviews/document-ingestion2-impl.md Pass 3, Blocker 1).
#
# Only rows where the property is genuinely absent are touched — a Document/
# Chunk already flipped `currentVersion`/`documentCurrent = false` (a real
# supersede) is left exactly as it is. Idempotent — a second run reports 0
# for both counts.
#
# Query (per docs/reviews/document-ingestion2-impl.md Pass 3's suggested fix),
# workspace-wide, two independent statements — the Document and Chunk
# backfills don't need to share a MATCH; every Chunk already carries its own
# `documentCurrent`, no traversal through Document required:
#
#   MATCH (d:Document) WHERE d.currentVersion IS NULL
#     SET d.currentVersion = true RETURN count(d) AS backfilled
#   MATCH (c:Chunk) WHERE c.documentCurrent IS NULL
#     SET c.documentCurrent = true RETURN count(c) AS backfilled
#
# Env vars:
#   FALKORDB_HOST  (default: 127.0.0.1)
#   FALKORDB_PORT  (default: 6379)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <workspaceId> [<workspaceId> ...]" >&2
  exit 1
fi

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}
echo "OK"

for wid in "$@"; do
  g="ws:${wid}"
  echo ""
  echo "── backfilling ${g} ─────────────────────────────────────"

  doc_out=$(redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$g" \
    "MATCH (d:Document) WHERE d.currentVersion IS NULL SET d.currentVersion = true RETURN count(d) AS backfilled")
  echo "$doc_out"
  doc_count=$(echo "$doc_out" | sed -n '2p' | tr -d '[:space:]')
  echo "backfilled ${doc_count:-?} document(s) in ${g}"

  chunk_out=$(redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$g" \
    "MATCH (c:Chunk) WHERE c.documentCurrent IS NULL SET c.documentCurrent = true RETURN count(c) AS backfilled")
  echo "$chunk_out"
  chunk_count=$(echo "$chunk_out" | sed -n '2p' | tr -d '[:space:]')
  echo "backfilled ${chunk_count:-?} chunk(s) in ${g}"
done

echo ""
echo "Backfill complete (idempotent — re-running reports 0)."
