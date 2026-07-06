#!/usr/bin/env bash
set -euo pipefail

# K-011 M1 DoD closeout harness: load-test the REST append path + GRAPH.PROFILE
# the four hot reads + capture a per-workspace RAM delta from INFO memory.
#
# Isolated to a throwaway `ws:load` workspace — never touches ws:acme/reference.
# Runs the whole flow end to end and prints a report; captures raw GRAPH.PROFILE
# output for the four hot reads to $OUT_DIR.
#
#   ./scripts/load_test.sh                 # defaults: 3000 msgs / 16 workers
#   LOAD_MESSAGES=5000 LOAD_WORKERS=32 ./scripts/load_test.sh
#   KEEP_WS=1 ./scripts/load_test.sh       # skip the ws:load teardown at the end
#
# Env overrides:
#   LOAD_MESSAGES (3000)  LOAD_WORKERS (16)  LOAD_WS (load)
#   FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)  SERVER_PORT (8100)
#   OUT_DIR (scripts/.load-out)  KEEP_WS (unset → tear down ws:load)

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

LOAD_WS="${LOAD_WS:-load}"
LOAD_MESSAGES="${LOAD_MESSAGES:-3000}"
LOAD_WORKERS="${LOAD_WORKERS:-16}"
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
SERVER_PORT="${SERVER_PORT:-8100}"   # off the default 8000 to avoid a stray dev server

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV="$SERVER_DIR/.venv"
PY="$VENV/bin/python"
OUT_DIR="${OUT_DIR:-$REPO_DIR/scripts/.load-out}"
BASE_URL="http://127.0.0.1:${SERVER_PORT}"
GRAPH="ws:${LOAD_WS}"

RCLI=(redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT")

log() { printf '\n=== %s ===\n' "$*"; }
mem() { "${RCLI[@]}" INFO memory | tr -d '\r' | awk -F: '/^used_memory:/ {print $2}'; }

# ── preconditions ───────────────────────────────────────────────────────────
"${RCLI[@]}" ping 2>/dev/null | grep -q PONG || {
  echo "ERROR: FalkorDB not reachable at $FALKORDB_HOST:$FALKORDB_PORT — start it with ./scripts/start_falkordb.sh -d" >&2
  exit 1
}
[ -x "$PY" ] || { echo "ERROR: venv missing — run: cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2; exit 1; }

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.json"

# ── 1. schema for ws:load ───────────────────────────────────────────────────
log "Bootstrapping schema for $GRAPH"
"$REPO_DIR/scripts/bootstrap_schema.sh" "$LOAD_WS" >/dev/null
MEM_BEFORE="$(mem)"
echo "used_memory after bootstrap, before load: $MEM_BEFORE bytes"

# ── 2. start the M1 server against ws:load ──────────────────────────────────
log "Starting server on $BASE_URL (FALKORCHAT_WS_ID=$LOAD_WS)"
SERVER_LOG="$OUT_DIR/server.log"
FALKORCHAT_WS_ID="$LOAD_WS" FALKORCHAT_USER_ID="u1" \
  FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$VENV/bin/uvicorn" --app-dir "$SERVER_DIR" \
  --host 127.0.0.1 --port "$SERVER_PORT" falkorchat.app:app \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# wait for /health
for i in $(seq 1 60); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then echo "server ready"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "ERROR: server exited early — see $SERVER_LOG" >&2; tail -20 "$SERVER_LOG" >&2; exit 1; fi
  [ "$i" -eq 60 ] && { echo "ERROR: server did not become healthy — see $SERVER_LOG" >&2; exit 1; }
  sleep 0.5
done

# ── 3. run the load harness ─────────────────────────────────────────────────
log "Load: $LOAD_MESSAGES messages / $LOAD_WORKERS workers"
LOAD_EMIT="$SUMMARY" "$PY" "$REPO_DIR/scripts/load_append.py" \
  --base-url "$BASE_URL" --workers "$LOAD_WORKERS" --messages "$LOAD_MESSAGES" \
  --emit "$SUMMARY"

MEM_AFTER="$(mem)"

# ── 4. RAM delta ────────────────────────────────────────────────────────────
log "RAM delta (INFO memory used_memory)"
MSGS_OK="$("$PY" -c "import json;print(json.load(open('$SUMMARY'))['messages_ok'])")"
THREAD_ID="$("$PY" -c "import json;print(json.load(open('$SUMMARY'))['sample_thread_id'])")"
"$PY" - "$MEM_BEFORE" "$MEM_AFTER" "$MSGS_OK" <<'PY'
import sys
before, after, msgs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
delta = after - before
print(f"used_memory before : {before:>14,} bytes")
print(f"used_memory after  : {after:>14,} bytes")
print(f"delta              : {delta:>14,} bytes  over {msgs:,} messages")
print(f"per message        : {delta/msgs:>14,.0f} bytes  (~{delta/msgs/1024:.2f} KB, chat-core, no embeddings)")
print(f"per 100k-msg ws    : {delta/msgs*100_000/1024/1024:>14,.0f} MB")
PY
echo
echo "GRAPH.MEMORY USAGE $GRAPH (cross-check — under-reports per §11 caveat):"
"${RCLI[@]}" GRAPH.MEMORY USAGE "$GRAPH" || true

# ── 5. GRAPH.PROFILE the four hot reads ─────────────────────────────────────
log "GRAPH.PROFILE the four hot reads (raw output → $OUT_DIR)"

profile() {  # $1 = label/file  $2 = query
  local name="$1" q="$2" f="$OUT_DIR/profile_$1.txt"
  {
    echo "### $name"
    echo "GRAPH.PROFILE $GRAPH \"$q\""
    echo "---"
    "${RCLI[@]}" GRAPH.PROFILE "$GRAPH" "$q"
  } | tee "$f"
  echo
}

Q4="MATCH (t:Thread {threadId: '$THREAD_ID'})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId, m.text, m.role, m.createdAt, coalesce(author.userId, author.agentId) AS authorId, author.displayName, labels(author) AS authorType ORDER BY m.createdAt"

Q91="MATCH (t:Thread {threadId: '$THREAD_ID'})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.createdAt > 0 OR (m.createdAt = 0 AND m.msgId > '') MATCH (m)-[:POSTED_BY]->(author) OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) WHERE me.userId = 'u1' OR me.agentId = 'u1' WITH m, author, count(me) > 0 AS isMention RETURN m.msgId, m.text, m.role, m.createdAt, coalesce(author.userId, author.agentId) AS authorId, labels(author) AS authorType, isMention, m.threadId AS threadId ORDER BY m.createdAt, m.msgId LIMIT 50"

Q92="MATCH (m:Message) WHERE m.createdAt > 0 OR (m.createdAt = 0 AND m.msgId > '') MATCH (m)-[:POSTED_BY]->(author) OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) WHERE me.userId = 'u1' OR me.agentId = 'u1' WITH m, author, count(me) > 0 AS isMention RETURN m.msgId, m.text, m.role, m.createdAt, coalesce(author.userId, author.agentId) AS authorId, labels(author) AS authorType, isMention, m.threadId AS threadId ORDER BY m.createdAt, m.msgId LIMIT 50"

Q5="CALL db.idx.fulltext.queryNodes('Message', 'load') YIELD node AS m, score RETURN m.msgId, m.threadId AS threadId, m.text, m.createdAt, score ORDER BY score DESC LIMIT 50"

profile "4_thread_read"     "$Q4"
profile "9_1_since_thread"  "$Q91"
profile "9_2_since_wswide"  "$Q92"
profile "5_search"          "$Q5"

# ── 6. scan-type check ──────────────────────────────────────────────────────
log "Scan-type check (want: 'Node By Index Scan'; flag: 'Node By Label Scan')"
for f in "$OUT_DIR"/profile_*.txt; do
  name="$(basename "$f" .txt)"
  if grep -qi 'Label Scan' "$f"; then
    echo "  [DEGRADED] $name — Node By Label Scan present (ESCALATE to graph-dba):"
    grep -i 'Scan' "$f" | sed 's/^/      /'
  else
    # Full-text §5 is anchored on the RediSearch full-text index via
    # ProcedureCall (db.idx.fulltext.queryNodes), not a node index scan — that
    # is the correct index-backed plan for it, not a degradation.
    echo "  [OK] $name — index-backed anchor:"
    grep -iE 'Index Scan|ProcedureCall|Procedure Call' "$f" | head -3 | sed 's/^/      /'
  fi
done

# ── 7. teardown ─────────────────────────────────────────────────────────────
cleanup; trap - EXIT
if [ -n "${KEEP_WS:-}" ]; then
  log "KEEP_WS set — leaving $GRAPH in place"
else
  log "Tearing down $GRAPH (the throwaway workspace this harness created)"
  "${RCLI[@]}" GRAPH.DELETE "$GRAPH"
fi

log "Done. Raw output in $OUT_DIR (summary.json, profile_*.txt, server.log)"
