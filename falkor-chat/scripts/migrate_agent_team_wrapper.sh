#!/usr/bin/env bash
# migrate_agent_team_wrapper.sh — the single lock-holding wrapper for the
# ws:agent-team embedder cutover (docs/plans/python-web-quirks-kb-prod-migration.md
# §4 "Step 3-5"). Rehearsed by scripts/rehearse_agent_team_wrapper.sh against stand-ins.
#
# WHY: claude/teco/hooks/ensure-environment.sh relaunches start_agent_team.sh (at the
# script's default EMBEDDING_DIM) whenever :PORT/health refuses the connection AND
# `flock -n -E 75 <lock> true` does not return 75. The lock is held by the running
# server for its lifetime, so stopping the server frees it. This wrapper is started
# as the *queued* next holder, so the lock never becomes free from the stop until the
# new server is serving (or forever, on a failure: see HOLD).
#
# INVOCATION (queue the waiter FIRST, then end the old holder, in the same shell command,
# so the lock is never free; -w 60 waits for the old server's graceful shutdown):
#   ( setsid nohup flock -w 60 <LOCK> bash <this-script> \
#       >>"$XDG_STATE_HOME/graphmind-ai-lab/agent-team-8200.log" 2>&1 </dev/null & ) ; \
#   kill <old-server-pid>
# flock(1) stays as this script's parent for its whole life and holds the lock itself;
# every descendant (stage commands, the exec'd server, `sleep infinity`) also inherits the
# fd. So the lock frees only when the wrapper/server/sleeper exits (the parent flock then
# exits too). No second flock is ever taken here. To end a HOLD, kill the sleeper (this
# script's pid, printed in the HOLD log line), never the flock parent.
#
# STAGES (all run with the lock fd inherited):
#   1. WRAP_STAGE1_CMD   backups (GRAPH.COPY, logical export, counts). REQUIRED, no default.
#      then WAIT for the go-file (WRAP_GO_FILE; WRAP_GO_TIMEOUT seconds, default 1800).
#   2. WRAP_STAGE2_CMD   migrate (default: migrate_embeddings.sh agent-team <4b> ...).
#   3. exec env -i <KEEP_ENV that is set> FALKORCHAT_WS_ID FALKORDB_HOST FALKORDB_PORT
#      AGENT_TEAM_PORT [WRAP_EXTRA_ENV] <WRAP_SERVER_CMD>   from WRAP_FC_DIR.
#      exec keeps the lock fd on the server for its life, exactly as today.
# HOLD: any failure or timeout in any stage runs `exec sleep infinity` (lock kept, hook
#   stays blocked) and logs "HOLD". Recovery is plan §5: queue the next `flock -w 60`
#   waiter first, then kill the sleeper (the handoff keeps the lock held throughout).
#
# VERIFY THAT THE WRAPPER (not merely somebody) HOLDS THE LOCK, before any copy:
#   WRAP_LOCK=... WRAP_STATE_DIR=... migrate_agent_team_wrapper.sh --check-held
#   exit 0 = wrapper pid alive, has the lock file open, and `flock -n -E 75` returns 75.
# A bare `flock -n -E 75 <lock> true` == 75 is NOT sufficient: if `flock -w` timed out
# while the old holder was still shutting down, that returns 75 too, and the lock is
# free (to the hook) the moment the old holder exits (observed in the rehearsal, a7).
#
# Parameters (environment; every default mirrors the hook's own derivation):
#   WRAP_PORT        AGENT_TEAM_PORT served            (8200)
#   WRAP_LOCK        start lock   ($XDG_STATE_HOME|~/.local/state)/graphmind-ai-lab/agent-team-$WRAP_PORT.lock
#   WRAP_STATE_DIR   go-file, events, stage logs       (.../graphmind-ai-lab/agent-team-migrate-$WRAP_PORT)
#   WRAP_GO_FILE     (\$WRAP_STATE_DIR/go)   WRAP_GO_TIMEOUT (1800 s)
#   WRAP_FC_DIR      cwd for stage 3 and default stage 2   (this script's falkor-chat/)
#   WRAP_SERVER_CMD  (\$WRAP_FC_DIR/scripts/start_agent_team.sh)
#   WRAP_STAGE1_CMD  required      WRAP_STAGE2_CMD  (migrate_embeddings.sh, see code)
#   WRAP_WS_ID (agent-team)  WRAP_FALKORDB_HOST (127.0.0.1)  WRAP_FALKORDB_PORT (6379)
#   WRAP_EXTRA_ENV   extra "K=V K=V" for the server (R1/R2: EMBEDDING_DIM=1024)
# Events (epoch-ms, one per line) go to $WRAP_STATE_DIR/wrapper.events; pid to wrapper.pid.

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

set -u
shopt -s execfail

PORT="${WRAP_PORT:-8200}"
STATE_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/graphmind-ai-lab"
LOCK="${WRAP_LOCK:-$STATE_ROOT/agent-team-$PORT.lock}"
STATE="${WRAP_STATE_DIR:-$STATE_ROOT/agent-team-migrate-$PORT}"
GO_FILE="${WRAP_GO_FILE:-$STATE/go}"
GO_TIMEOUT="${WRAP_GO_TIMEOUT:-1800}"
FC_DIR="${WRAP_FC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SERVER_CMD="${WRAP_SERVER_CMD:-$FC_DIR/scripts/start_agent_team.sh}"
WS_ID="${WRAP_WS_ID:-agent-team}"
FDB_HOST="${WRAP_FALKORDB_HOST:-127.0.0.1}"
FDB_PORT="${WRAP_FALKORDB_PORT:-6379}"
EXTRA_ENV="${WRAP_EXTRA_ENV:-}"
STAGE1_CMD="${WRAP_STAGE1_CMD:-}"
STAGE2_CMD="${WRAP_STAGE2_CMD:-cd \"$FC_DIR\" && FALKORCHAT_OPENCODE_CONFIG=\$HOME/.config/opencode/opencode.local.json ./scripts/migrate_embeddings.sh agent-team lmstudio/text-embedding-qwen3-embedding-4b --i-have-stopped-traffic}"

mkdir -p "$STATE" || { echo "wrapper: cannot create $STATE" >&2; exit 2; }

if [ "${1:-}" = "--check-held" ]; then
  pid="$(cat "$STATE/wrapper.pid" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "NOT HELD: no live wrapper pid in $STATE/wrapper.pid"; exit 1
  fi
  want="$(stat -c '%d:%i' "$LOCK" 2>/dev/null || true)"
  has=0
  for f in /proc/"$pid"/fd/*; do
    [ -n "$want" ] && [ "$(stat -L -c '%d:%i' "$f" 2>/dev/null)" = "$want" ] && has=1
  done
  [ "$has" -eq 1 ] || { echo "NOT HELD: wrapper pid $pid has no fd on $LOCK"; exit 1; }
  flock -n -E 75 "$LOCK" true; rc=$?
  [ "$rc" -eq 75 ] || { echo "NOT HELD: flock -n -E 75 returned $rc"; exit 1; }
  echo "HELD by wrapper pid $pid"; exit 0
fi

ms() { local t="${EPOCHREALTIME/./}"; echo "${t%???}"; }
ev() { printf '%s %s\n' "$(ms)" "$*" >>"$STATE/wrapper.events"; echo "$(date -Is) wrapper: $*"; }
# Keep the lock forever (exec keeps pid and fds). The loop is only a fallback if exec fails.
hold() { ev "HOLD pid=$$: $*"; exec sleep infinity; while :; do sleep 3600; done; }

: >"$STATE/wrapper.events"
echo "$$" >"$STATE/wrapper.pid"

# Refuse to do anything unless the lock is really held (run without `flock <lock>`, or
# after a flock timeout, nothing protects the window). Holds nothing, so it exits.
flock -n -E 75 "$LOCK" true; rc=$?
if [ "$rc" -ne 75 ]; then ev "LOCK NOT HELD (flock -n returned $rc); refusing to run"; exit 3; fi
ev "LOCK_ACQUIRED pid=$$ lock=$LOCK"

# ── stage 1: backups, then wait for the go-file ────────────────────────────────
[ -n "$STAGE1_CMD" ] || hold "WRAP_STAGE1_CMD is not set"
ev "STAGE1_START"
bash -c "$STAGE1_CMD" >>"$STATE/stage1.log" 2>&1; rc=$?
[ "$rc" -eq 0 ] || hold "stage 1 failed (rc=$rc)"
ev "STAGE1_END"

ev "GO_WAIT_START $GO_FILE"
deadline=$(( SECONDS + GO_TIMEOUT ))
while [ ! -e "$GO_FILE" ]; do
  [ "$SECONDS" -lt "$deadline" ] || hold "go-file timeout after ${GO_TIMEOUT}s"
  sleep 0.2
done
ev "GO_SEEN"

# ── stage 2: migrate ───────────────────────────────────────────────────────────
ev "STAGE2_START"
bash -c "$STAGE2_CMD" >>"$STATE/stage2.log" 2>&1; rc=$?
ev "STAGE2_END rc=$rc"
[ "$rc" -eq 0 ] || hold "stage 2 failed (rc=$rc)"

# ── stage 3: serve (same env construction as ensure-environment.sh:111-117) ─────
KEEP_ENV="PATH HOME USER LOGNAME LANG LC_ALL TMPDIR XDG_RUNTIME_DIR XDG_CONFIG_HOME XDG_CACHE_HOME
  DOCKER_HOST DOCKER_CONTEXT DOCKER_CONFIG
  http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY SSL_CERT_FILE SSL_CERT_DIR"
kept_env=()
for v in $KEEP_ENV; do
  [ -n "${!v+x}" ] && kept_env+=("$v=${!v}")
done
extra=()
for kv in $EXTRA_ENV; do extra+=("$kv"); done

[ -x "$SERVER_CMD" ] || hold "server command not executable: $SERVER_CMD"
[ -d "$FC_DIR" ] || hold "WRAP_FC_DIR missing: $FC_DIR"
ev "EXEC $SERVER_CMD"
cd "$FC_DIR" && exec env -i ${kept_env[@]+"${kept_env[@]}"} FALKORCHAT_WS_ID="$WS_ID" FALKORDB_HOST="$FDB_HOST" \
  FALKORDB_PORT="$FDB_PORT" AGENT_TEAM_PORT="$PORT" ${extra[@]+"${extra[@]}"} "$SERVER_CMD"
hold "exec of the server failed"
