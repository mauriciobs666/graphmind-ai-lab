#!/usr/bin/env bash
# backfill_thread_ids.sh — one-off: stamp Message.threadId on pre-K-007 messages.
#
# Usage:
#   ./scripts/backfill_thread_ids.sh <workspaceId> [<workspaceId> ...]
#
# The K-007 v2 write paths stamp `threadId` inline on every new message; this
# script backfills messages written before that. Run once per existing
# workspace after deploying v2. Idempotent — a second run reports 0.
#
# Query (canonical: docs/QUERIES.md §4.x), workspace-wide variant:
#
#   MATCH (t:Thread)-[:HEAD]->(first:Message)
#   MATCH (first)-[:NEXT*0..]->(m:Message)
#   WHERE m.threadId IS NULL
#   SET m.threadId = t.threadId
#   RETURN count(m) AS backfilled
#
# Large workspaces: writes CANNOT be killed by TIMEOUT on this build — bound
# the work yourself by running the per-thread batched variant instead (add
# `{threadId: $threadId}` to the Thread anchor and loop over thread ids):
#
#   MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
#   MATCH (first)-[:NEXT*0..]->(m:Message)
#   WHERE m.threadId IS NULL
#   SET m.threadId = t.threadId
#   RETURN count(m) AS backfilled
#
# Orphan caveat: the walk anchors on HEAD, so messages unreachable from a HEAD
# (residue of the pre-v2 write defects) are NOT backfilled — acceptable, they
# are already invisible to thread reads. Until backfilled, old rows return
# threadId: null in §9.2/§5 results (clients must tolerate null).
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
  out=$(redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$g" \
    "MATCH (t:Thread)-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.threadId IS NULL SET m.threadId = t.threadId RETURN count(m) AS backfilled")
  echo "$out"
  count=$(echo "$out" | sed -n '2p' | tr -d '[:space:]')
  echo "backfilled ${count:-?} message(s) in ${g}"
done

echo ""
echo "Backfill complete (idempotent — re-running reports 0)."
