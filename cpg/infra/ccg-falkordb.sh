#!/usr/bin/env bash
set -euo pipefail
# ccg-falkordb.sh — the dedicated FalkorDB instance for C-1001 (`ccg-falkordb`, 6390), plus
# disposable per-step instances (`--throwaway`) for live tests that mutate ACL users or
# GRAPH.CONFIG, or run in parallel.
#
# Design: docs/plans/connected-code-graph-p0.md §2 D7, §5.1; P1 step P1-48
# (docs/plans/connected-code-graph-p1-extraction.md). Instance flags: -p0-graph.md §6 rows 5, 12.
# Regression test (stubbed docker/redis-cli, never real containers): tests/test-ccg-falkordb.sh
#
#   up        start the instance if it is not running (idempotent; a running instance is a no-op),
#             set the P1 module config, and report TIMEOUT_MAX / MAX_INFO_QUERIES from GRAPH.CONFIG GET
#   status    print state, binding, the D7 config, image/module, memory cap, P1 steady-state
#             hygiene and key count (read-only; non-zero exit if the instance is down or drifted)
#   down      stop and remove the container (refuses if its keyspace is non-empty unless
#             --force: keys there may belong to another step's reservation, §5.1)
#   reset-p0  FLUSHALL the keyspace of ccg-falkordb (the name is from P0, where it ran at exit;
#             it is the general approval-gated keyspace reset). The ONLY destructive verb on the
#             shared instance; refuses unless the container is ccg-falkordb on 127.0.0.1:6390,
#             and requires --yes. Approval-gated. Refused in --throwaway mode.
#
#   --throwaway [up]   start a DISPOSABLE instance on a free port (CCG_FALKORDB_PORT, or the first
#                      free port in 16390-16489), named ccg-falkordb-tw-<port>, labelled
#                      ccg-falkordb=1 and ccg-falkordb.throwaway=1. Refuses 6379, 6390 and any
#                      port already published or listening. Prints CCG_FALKORDB_NAME=… and
#                      CCG_FALKORDB_PORT=… on stdout (logs go to stderr).
#   --throwaway status|down    same verbs for the throwaway named by CCG_FALKORDB_PORT; down needs
#                      no --force (the keyspace belongs to the caller that started it).
#   --throwaway list   list throwaway containers (name, state, ports).
#
# Invariants:
#  - OWNERSHIP: no `docker stop`/`docker rm` and no Redis write ever reaches a container that does
#    not carry the label ccg-falkordb=1, which only this script's `docker run` sets. That is what
#    makes a CCG_FALKORDB_NAME override safe: pointing it at `falkordb-dev` (unlabelled) is
#    refused by every verb. The shared `falkordb-dev` (6379) is never stopped, restarted,
#    reconfigured or written to by anything here. --throwaway verbs further require the
#    ccg-falkordb.throwaway=1 label, so they can never stop or remove `ccg-falkordb` itself.
#  - The key-count guard and every Redis call go to the port the target container itself
#    publishes (`docker port`), never to a $CCG_FALKORDB_PORT that might name another instance.
#  - Binds 127.0.0.1 only. `falkordb-dev` publishes 0.0.0.0, so do not copy its `-p` form.
#  - The admin password lives in ~/.config/ccg/falkordb-admin.pass (mode 600, outside the repo),
#    created on first `up` if absent, never printed, never on a command line, never in a
#    container env var (`docker inspect` would show it). It reaches redis-server through a
#    generated config file mounted read-only, and reaches redis-cli through REDISCLI_AUTH.
#    Throwaways share that password file.
#  - No persistence (save "", appendonly no), so there is no volume: a removed container loses
#    nothing that was not already lost the moment it stopped, and the first query after a
#    restart is a rebuild. `down --force` + `up` is therefore the full clean start (keyspace, ACL
#    users and runtime config all reset).
#  - P1 module config (TIMEOUT_MAX, MAX_INFO_QUERIES) is set at runtime by `up` with GRAPH.CONFIG
#    SET right after the container starts; FALKORDB_ARGS is never passed, so the image's other
#    module defaults stay as they are on falkordb-dev. An instance restarted behind this script's
#    back loses them, and `status` reports that as drift.
#
# Env overrides (defaults are D7's; reset-p0 still refuses anything but the defaults, and status
# compares the image against the pinned default, not against the override):
#   CCG_FALKORDB_NAME   container name          (ccg-falkordb; with --throwaway it must equal
#                                                ccg-falkordb-tw-<port> if set at all)
#   CCG_FALKORDB_PORT   host port on 127.0.0.1  (6390; with --throwaway: chosen if unset for up,
#                                                required for status/down)
#   CCG_FALKORDB_IMAGE  image                   (falkordb/falkordb:v4.18.11 — pinned, K-P0-9)
#   CCG_CONFIG_DIR      password/config dir     (~/.config/ccg)

PINNED_IMAGE="falkordb/falkordb:v4.18.11"
PINNED_MODULE_VER="41811"            # graph module version shipped in PINNED_IMAGE
SHARED_NAME="ccg-falkordb"
SHARED_PORT=6390
DEV_PORT=6379                        # falkordb-dev: never a target
TW_PREFIX="ccg-falkordb-tw-"
TW_RANGE_LO=16390
TW_RANGE_HI=16489

THROWAWAY=0
ARGS=()
for a in "$@"; do
  if [ "$a" = --throwaway ]; then THROWAWAY=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

NAME_ENV="${CCG_FALKORDB_NAME:-}"
PORT_ENV="${CCG_FALKORDB_PORT:-}"
if [ "$THROWAWAY" -eq 1 ]; then
  # Leading zeros are stripped at any length (0006390 is 6390), so the refusals and the derived name see
  # one spelling; tw_check_port_value then accepts only ^[1-9][0-9]{3,4}$ in 1024-65535.
  case "$PORT_ENV" in ''|*[!0-9]*) ;; *) PORT_ENV="${PORT_ENV#"${PORT_ENV%%[!0]*}"}"; PORT_ENV="${PORT_ENV:-0}" ;; esac
  PORT="$PORT_ENV"
  NAME="${PORT:+$TW_PREFIX$PORT}"
else
  NAME="${NAME_ENV:-$SHARED_NAME}"
  PORT="${PORT_ENV:-$SHARED_PORT}"
fi
IMAGE="${CCG_FALKORDB_IMAGE:-$PINNED_IMAGE}"
CFG_DIR="${CCG_CONFIG_DIR:-$HOME/.config/ccg}"
PASS_FILE="$CFG_DIR/falkordb-admin.pass"
HOST=127.0.0.1
LABEL_KEY="ccg-falkordb"
TW_LABEL_KEY="ccg-falkordb.throwaway"
CLI_PORT="$PORT"                     # re-pointed at the container's own published port
conf_file() { echo "$CFG_DIR/$NAME.redis.conf"; }

# D7 config, the single source both `up` (writes) and `status` (verifies) read.
MAXMEMORY="1536mb"
MAXMEMORY_BYTES=$((1536 * 1024 * 1024))
POLICY="noeviction"
# Container memory cap. maxmemory bounds only Redis's accounted used_memory, checked when a
# command starts; allocator fragmentation, client buffers and one admitted query's overshoot
# are outside it. 2g = maxmemory + 512 MiB (~1.33x) covers those, and keeps a runaway inside
# this container instead of the 6 GB VM's OOM killer, which could take falkordb-dev or a timed
# Joern JVM. memory-swap == memory means no swap: a would-be-swapping instance (silently
# distorted S20 timings) becomes a loud OOM kill instead, which costs nothing here because
# nothing persists.
MEM_CAP="2g"
MEM_CAP_BYTES=$((2 * 1024 * 1024 * 1024))
# P1 module config (-p0-graph.md §6): row 5 TIMEOUT_MAX default 10,000 ms (P1-19 re-derives it,
# so `status` accepts any nonzero value); row 12 MAX_INFO_QUERIES 0 (no telemetry stream).
TIMEOUT_MAX_UP=10000
MAX_INFO_QUERIES_UP=0
# The P1 steady-state ACL user set (P1-19's create_users): default plus, per system <s>, the four
# users ccg_<role>_<s> with role build|verdict|engine|raw; any other user is drift. <s> follows the
# system-name whitelist of docs/plans/connected-code-graph-p1-graph.md §7 (^[a-z0-9]+(?:[-_][a-z0-9]+)*$,
# length 1-48). Kept local until ccg.sysname (its home) exists.
ACL_USER_RE='^ccg_(build|verdict|engine|raw)_([a-z0-9]+([-_][a-z0-9]+)*)$'
acl_user_ok() { # <user>
  [ "$1" = default ] && return 0
  [[ $1 =~ $ACL_USER_RE ]] || return 1
  [ "${#BASH_REMATCH[2]}" -ge 1 ] && [ "${#BASH_REMATCH[2]}" -le 48 ]
}

log() { echo "ccg-falkordb.sh: $*" >&2; }
die() { log "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: ccg-falkordb.sh <up|status|down [--force]|reset-p0 --yes|-h>
       ccg-falkordb.sh --throwaway [up|status|down|list]

  up        start $SHARED_NAME on $HOST:$SHARED_PORT if not running (idempotent); sets and reports
            TIMEOUT_MAX $TIMEOUT_MAX_UP and MAX_INFO_QUERIES $MAX_INFO_QUERIES_UP
  status    report state + D7 config + P1 hygiene; non-zero exit if down or drifted
  down      stop and remove $SHARED_NAME (refuses a non-empty keyspace without --force)
  reset-p0  FLUSHALL the keyspace of $SHARED_NAME (needs --yes; approval-gated)

  --throwaway up      disposable instance on CCG_FALKORDB_PORT or the first free port in
                      $TW_RANGE_LO-$TW_RANGE_HI (never $DEV_PORT or $SHARED_PORT); prints
                      CCG_FALKORDB_NAME=… and CCG_FALKORDB_PORT=… on stdout
  --throwaway status  status of the throwaway on CCG_FALKORDB_PORT
  --throwaway down    remove the throwaway on CCG_FALKORDB_PORT (no --force needed)
  --throwaway list    list throwaway containers

Full clean start (keyspace, ACL users, runtime config): down --force, then up.
Connect:  REDISCLI_AUTH="\$(cat $PASS_FILE)" redis-cli -h $HOST -p <port>
EOF
}

preflight() {
  local t
  for t in docker redis-cli openssl ss; do
    command -v "$t" >/dev/null 2>&1 || die "$t not on PATH (redis-cli: redis-tools; ss: iproute2)"
  done
  docker info >/dev/null 2>&1 </dev/null || die "Docker daemon not reachable"
}

# container state: running | exited | created | ... | absent
state() {
  # `docker inspect` on a missing name still prints an empty line before failing, so the
  # output is captured and the fallback decided on the exit code, not appended to it.
  local s
  if s="$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null </dev/null)"; then
    echo "$s"
  else
    echo absent
  fi
}

label_of() { # [label key]
  docker inspect "$NAME" --format "{{index .Config.Labels \"${1:-$LABEL_KEY}\"}}" 2>/dev/null </dev/null || true
}

# The ownership gate. Called before every docker stop/rm and every Redis write.
own_container() {
  [ "$(label_of)" = 1 ] || die "refusing: container '$NAME' does not carry the label $LABEL_KEY=1, so this script did not create it (it never stops, removes or writes to a container it does not own)"
  if [ "$THROWAWAY" -eq 1 ]; then
    [ "$(label_of "$TW_LABEL_KEY")" = 1 ] || die "refusing: container '$NAME' does not carry the label $TW_LABEL_KEY=1, so it is not a throwaway"
  fi
}

# The host port the container itself publishes for 6379, if it is exactly one 127.0.0.1 line.
# Prints the port, or nothing (and returns 1) if the binding is missing/plural/non-loopback.
container_port() {
  local lines
  lines="$(docker port "$NAME" 6379/tcp 2>/dev/null </dev/null || true)"
  [ -n "$lines" ] && [ "$(printf '%s\n' "$lines" | wc -l)" -eq 1 ] || return 1
  case "$lines" in
    "$HOST":*) echo "${lines#"$HOST":}" ;;
    *) return 1 ;;
  esac
}

ensure_password() {
  (umask 077 && mkdir -p "$CFG_DIR")
  chmod 700 "$CFG_DIR"
  if [ ! -s "$PASS_FILE" ]; then
    local pw
    pw="$(openssl rand -hex 24)"            # generated first: a failure leaves no empty file
    [ -n "$pw" ] || die "openssl produced no password"
    (umask 077 && printf '%s\n' "$pw" >"$PASS_FILE")
    log "created admin password file $PASS_FILE (mode 600; not printed)"
  fi
  chmod 600 "$PASS_FILE"
}

# Regenerated on every `up` from the constants above + the password file, so the file can
# never drift from D7. The container runs as root, so a 600 file mounted :ro is readable.
write_conf() {
  local pass cf
  pass="$(cat "$PASS_FILE")"; cf="$(conf_file)"
  (umask 077 && cat >"$cf" <<EOF
# generated by cpg/infra/ccg-falkordb.sh — do not edit, it is rewritten on every \`up\`
requirepass $pass
maxmemory $MAXMEMORY
maxmemory-policy $POLICY
save ""
appendonly no
EOF
  )
  chmod 600 "$cf"
}

cli() {
  [ -s "$PASS_FILE" ] || die "no password file at $PASS_FILE (run: $0 up)"
  REDISCLI_AUTH="$(cat "$PASS_FILE")" redis-cli -h "$HOST" -p "$CLI_PORT" --no-auth-warning "$@" </dev/null
}

# Total keys across every db, from INFO keyspace. Prints a number, or "?" if the reply is not
# a keyspace section (NOAUTH, connection error, ...). redis-cli exits 0 on an error reply, so
# the reply's shape is what is checked, not the exit code.
key_count() {
  local out
  out="$(cli INFO keyspace 2>/dev/null | tr -d '\r' || true)"
  case "$out" in
    "# Keyspace"*) printf '%s\n' "$out" | sed -n 's/^db[0-9]*:keys=\([0-9]*\).*/\1/p' | awk '{s+=$1} END {print s+0}' ;;
    *) echo "?" ;;
  esac
}

wait_ready() {
  local i
  for i in $(seq 1 60); do
    if [ "$(cli PING 2>/dev/null || true)" = "PONG" ]; then return 0; fi
    sleep 0.5
  done
  die "$NAME did not answer an authenticated PING on $HOST:$CLI_PORT within 30 s (docker logs $NAME)"
}

graph_config_get() { cli GRAPH.CONFIG GET "$1" | sed -n 2p; }

# Applied only to a container this run just started (ownership is implied: our own docker run).
apply_module_config() {
  local k v r
  for kv in "TIMEOUT_MAX $TIMEOUT_MAX_UP" "MAX_INFO_QUERIES $MAX_INFO_QUERIES_UP"; do
    k="${kv% *}"; v="${kv#* }"
    r="$(cli GRAPH.CONFIG SET "$k" "$v")"
    [ "$r" = OK ] || die "GRAPH.CONFIG SET $k $v replied '$r'"
  done
}

module_config_ok() {
  local tmax miq
  tmax="$(graph_config_get TIMEOUT_MAX)"; miq="$(graph_config_get MAX_INFO_QUERIES)"
  case "$tmax" in ''|*[!0-9]*|0) return 1 ;; esac
  [ "$miq" = "$MAX_INFO_QUERIES_UP" ]
}

report_module_config() {
  local tmax miq
  tmax="$(graph_config_get TIMEOUT_MAX)"; miq="$(graph_config_get MAX_INFO_QUERIES)"
  log "GRAPH.CONFIG GET: TIMEOUT_MAX $tmax, MAX_INFO_QUERIES $miq"
}

port_listening() { # <port>
  ss -ltnH "sport = :$1" 2>/dev/null | grep -q .
}
port_in_use() { port_listening "$PORT"; }

# Published by any running container, on any address (docker ps .Ports: "0.0.0.0:6379->6379/tcp, …").
port_published() { # <port>
  docker ps --format '{{.Ports}}' </dev/null 2>/dev/null | tr ',' '\n' | grep -qE "(^|[^0-9])$1->"
}

# --throwaway's port gate. Pure argument checks first, so 6379/6390 are refused before any
# docker call.
tw_check_port_value() { # <port>
  [[ $1 =~ ^[1-9][0-9]{3,4}$ ]] || die "--throwaway: port '$1' is not a port number (1024-65535, no sign or spaces)"
  [ "$1" -ge 1024 ] && [ "$1" -le 65535 ] || die "--throwaway: port $1 is outside 1024-65535"
  [ "$1" != "$DEV_PORT" ]    || die "--throwaway refuses port $DEV_PORT (the shared falkordb-dev)"
  [ "$1" != "$SHARED_PORT" ] || die "--throwaway refuses port $SHARED_PORT (ccg-falkordb)"
}
tw_check_name() {
  [ -z "$NAME_ENV" ] || [ "$NAME_ENV" = "$NAME" ] \
    || die "--throwaway: CCG_FALKORDB_NAME='$NAME_ENV' is not $TW_PREFIX<port> for port ${PORT:-?}; throwaway names derive from the port"
}

cmd_up() {
  preflight
  local st p
  st="$(state)"
  case "$st" in
    running)
      own_container
      p="$(container_port)" || die "$NAME is running but not published on exactly one $HOST port (see: $0 status)"
      CLI_PORT="$p"
      log "$NAME already running — no-op"
      wait_ready
      report_module_config
      # (Only the shared instance gets here: --throwaway up refuses an existing name earlier.)
      module_config_ok || die "$NAME is running but its module config was lost (restarted behind up's back?); it holds no data, so: $0 down --force, then $0 up"
      return 0 ;;
    absent) ;;
    *)
      # Our own container, stopped (e.g. after a Docker/WSL restart). It holds no data
      # (no persistence), so removing it to recreate with current config loses nothing.
      own_container
      log "$NAME exists in state '$st' — removing the stopped container and recreating"
      docker rm "$NAME" >/dev/null </dev/null ;;
  esac
  port_in_use && die "port $PORT is already in use by something other than $NAME; refusing"
  run_container || die "docker run failed for $NAME on $HOST:$PORT"
}

# docker run of $NAME on $PORT, then readiness and the P1 module config. Returns 3 (and leaves
# nothing behind) if the container could not be created or started, so --throwaway can retry.
run_container() {
  local labels=(--label "$LABEL_KEY=1")
  [ "$THROWAWAY" -eq 1 ] && labels+=(--label "$TW_LABEL_KEY=1")
  ensure_password
  write_conf
  log "starting $NAME ($IMAGE) on $HOST:$PORT, memory cap $MEM_CAP (no swap)"
  # BROWSER=0: skip the bundled Node web console (saves RAM on the 6 GB host; no port 3000).
  # REDIS_ARGS: the image's run.sh puts it first on redis-server's argv, which is where a
  #   config-file path must go. FALKORDB_ARGS is deliberately NOT passed, so the image's
  #   module defaults stay as they are on falkordb-dev (bare `-e VAR` would delete them).
  # --pull=never: the pinned image is already local; a miss should say so, not pull.
  if ! docker run -d \
    --name "$NAME" \
    "${labels[@]}" \
    --pull=never \
    --memory "$MEM_CAP" --memory-swap "$MEM_CAP" \
    -p "$HOST:$PORT:6379" \
    -e BROWSER=0 \
    -e REDIS_ARGS=/etc/ccg/redis.conf \
    -v "$(conf_file):/etc/ccg/redis.conf:ro" \
    "$IMAGE" >/dev/null </dev/null; then
    # A failed start (e.g. the port was taken between the check and the run) can leave a
    # `created` container carrying our label; remove it so nothing is left behind.
    # Only a never-started (`created`) one: a name conflict means another run's container,
    # which must not be touched.
    if [ "$(state)" = created ] && [ "$(label_of)" = 1 ]; then docker rm "$NAME" >/dev/null 2>&1 </dev/null || true; fi
    return 3
  fi
  CLI_PORT="$PORT"
  # A throwaway that started but never became usable is removed (it is ours and disposable); the
  # shared instance is left for `status` to explain.
  if [ "$THROWAWAY" -eq 1 ]; then TW_STARTED="$NAME"; trap tw_cleanup_on_exit EXIT; fi
  wait_ready
  apply_module_config
  report_module_config
  TW_STARTED=""
  log "$NAME up"
}

TW_STARTED=""
tw_cleanup_on_exit() {
  local rc=$?
  if [ -n "$TW_STARTED" ] && [ "$(label_of "$TW_LABEL_KEY")" = 1 ]; then
    log "removing $TW_STARTED: it started but did not become usable"
    docker stop "$TW_STARTED" >/dev/null 2>&1 </dev/null || true
    docker rm "$TW_STARTED" >/dev/null 2>&1 </dev/null || true
    rm -f "$(conf_file)"
  fi
  exit "$rc"
}

cmd_tw_up() {
  local candidates=() p rc
  if [ -n "$PORT" ]; then
    tw_check_port_value "$PORT"
    tw_check_name
    candidates=("$PORT")
  else
    [ -z "$NAME_ENV" ] || die "--throwaway up: CCG_FALKORDB_NAME is set but CCG_FALKORDB_PORT is not; throwaway names derive from the port"
    for ((p = TW_RANGE_LO; p <= TW_RANGE_HI; p++)); do candidates+=("$p"); done
  fi
  preflight
  for p in "${candidates[@]}"; do
    PORT="$p"; NAME="$TW_PREFIX$p"; CLI_PORT="$p"
    if port_listening "$p" || port_published "$p"; then
      [ "${#candidates[@]}" -eq 1 ] && die "--throwaway refuses port $p: already published or listening"
      continue
    fi
    if [ "$(state)" != absent ]; then
      [ "${#candidates[@]}" -eq 1 ] && die "--throwaway: a container named $NAME already exists (remove it with: CCG_FALKORDB_PORT=$p $0 --throwaway down)"
      continue
    fi
    rc=0; run_container || rc=$?
    if [ "$rc" -eq 3 ]; then
      [ "${#candidates[@]}" -eq 1 ] && die "--throwaway: docker run failed for $NAME on $HOST:$p"
      log "docker run failed on $p; trying the next port"; continue
    fi
    echo "CCG_FALKORDB_NAME=$NAME"
    echo "CCG_FALKORDB_PORT=$p"
    return 0
  done
  die "--throwaway: no free port in $TW_RANGE_LO-$TW_RANGE_HI"
}

cmd_tw_need_port() {
  [ -n "$PORT" ] || die "--throwaway $1 needs CCG_FALKORDB_PORT (the port --throwaway up printed)"
  tw_check_port_value "$PORT"
  tw_check_name
}

cmd_tw_down() {
  cmd_tw_need_port down
  preflight
  local st
  st="$(state)"
  if [ "$st" = absent ]; then log "$NAME not present — no-op"; rm -f "$(conf_file)"; return 0; fi
  own_container
  if [ "$st" = running ]; then docker stop "$NAME" >/dev/null </dev/null; fi
  docker rm "$NAME" >/dev/null </dev/null
  rm -f "$(conf_file)"
  log "$NAME stopped and removed"
}

cmd_tw_list() {
  preflight
  docker ps -a --filter "label=$TW_LABEL_KEY=1" --format '{{.Names}}\t{{.State}}\t{{.Ports}}' </dev/null
}

cmd_status() {
  preflight
  local st rc=0 lab img ports nports p
  drift() { echo "DRIFT: $*"; rc=1; }
  st="$(state)"
  echo "container: $NAME"
  echo "state:     $st"
  if [ "$st" != running ]; then
    if [ "$THROWAWAY" -eq 1 ]; then echo "(not running — start one with: $0 --throwaway up)"; else echo "(not running — start with: $0 up)"; fi
    return 1
  fi
  lab="$(label_of)"
  img="$(docker inspect "$NAME" --format '{{.Config.Image}}' </dev/null)"
  ports="$(docker port "$NAME" </dev/null || true)"
  nports="$(printf '%s' "$ports" | grep -c . || true)"
  echo "label:     $LABEL_KEY=${lab:-<none>}"
  echo "image:     $img"
  echo "ports:     $(printf '%s' "$ports" | tr '\n' ' ') ($nports binding line(s))"
  [ "$lab" = 1 ]             || drift "label $LABEL_KEY=1 missing — this script did not create $NAME"
  if [ "$THROWAWAY" -eq 1 ]; then
    [ "$(label_of "$TW_LABEL_KEY")" = 1 ] || drift "label $TW_LABEL_KEY=1 missing — $NAME is not a throwaway"
  fi
  [ "$img" = "$PINNED_IMAGE" ] || drift "image is $img, not the pinned $PINNED_IMAGE"
  [ "$nports" = 1 ]          || drift "expected exactly 1 binding line, got $nports"
  [ "$ports" = "6379/tcp -> $HOST:$PORT" ] || drift "binding is not exactly $HOST:$PORT"
  if ! p="$(container_port)"; then
    drift "no single $HOST binding to query"; echo "config:    DRIFTED"; return 1
  fi
  CLI_PORT="$p"

  local mem memswap
  mem="$(docker inspect "$NAME" --format '{{.HostConfig.Memory}}' </dev/null)"
  memswap="$(docker inspect "$NAME" --format '{{.HostConfig.MemorySwap}}' </dev/null)"
  echo "memory cap:  $mem / swap+mem $memswap ($MEM_CAP = $MEM_CAP_BYTES, no swap)"
  [ "$mem" = "$MEM_CAP_BYTES" ]     || drift "memory cap"
  [ "$memswap" = "$MEM_CAP_BYTES" ] || drift "memory-swap cap"

  # Auth must be ON: an unauthenticated client has to be refused. redis-cli exits 0 either
  # way, so the reply text is what is checked.
  local unauth
  unauth="$(env -u REDISCLI_AUTH redis-cli -h "$HOST" -p "$CLI_PORT" PING 2>&1 </dev/null || true)"
  echo "unauth PING: $unauth"
  case "$unauth" in NOAUTH*) ;; *) drift "an unauthenticated PING was not refused (no requirepass?)" ;; esac
  if [ "$(cli PING 2>/dev/null || true)" != PONG ]; then
    drift "authenticated PING failed (wrong password file $PASS_FILE?)"
    echo "config:    DRIFTED from D7"; return 1
  fi

  local mm pol save aof modver tmax miq acl u bad_users="" keys
  mm="$(cli CONFIG GET maxmemory | sed -n 2p)"
  pol="$(cli CONFIG GET maxmemory-policy | sed -n 2p)"
  save="$(cli CONFIG GET save | sed -n 2p)"
  aof="$(cli CONFIG GET appendonly | sed -n 2p)"
  echo "maxmemory:        $mm ($MAXMEMORY = $MAXMEMORY_BYTES)"
  echo "maxmemory-policy: $pol"
  echo "save:             \"$save\""
  echo "appendonly:       $aof"
  [ "$mm" = "$MAXMEMORY_BYTES" ] || drift "maxmemory"
  [ "$pol" = "$POLICY" ]         || drift "maxmemory-policy"
  [ -z "$save" ]                 || drift "save"
  [ "$aof" = "no" ]              || drift "appendonly"
  modver="$(cli MODULE LIST | awk 'p=="name" && $0=="graph" {g=1} g && p=="ver" {print; exit} {p=$0}')"
  echo "module:    graph ver ${modver:-<none>} (pinned $PINNED_MODULE_VER)"
  [ "$modver" = "$PINNED_MODULE_VER" ] || drift "graph module version"
  # P1 steady state: a nonzero TIMEOUT_MAX (up sets $TIMEOUT_MAX_UP; P1-19 re-derives it),
  # MAX_INFO_QUERIES 0, and no ACL user outside default + the four ccg_* users.
  tmax="$(graph_config_get TIMEOUT_MAX)"
  miq="$(graph_config_get MAX_INFO_QUERIES)"
  acl="$(cli ACL USERS | paste -sd' ' -)"
  echo "TIMEOUT_MAX:      $tmax (up sets $TIMEOUT_MAX_UP; must be nonzero)"
  echo "MAX_INFO_QUERIES: $miq (must be $MAX_INFO_QUERIES_UP)"
  echo "ACL users:   $acl (allowed: default, ccg_{build,verdict,engine,raw}_<system>)"
  case "$tmax" in ''|*[!0-9]*|0) drift "GRAPH.CONFIG TIMEOUT_MAX is '${tmax}', not a nonzero value (restarted behind up's back?)" ;; esac
  [ "$miq" = "$MAX_INFO_QUERIES_UP" ] || drift "GRAPH.CONFIG MAX_INFO_QUERIES is '$miq', not $MAX_INFO_QUERIES_UP"
  for u in $acl; do
    acl_user_ok "$u" || bad_users+="$u "
  done
  [ -z "$bad_users" ] || drift "ACL users outside the P1 set: ${bad_users% }"
  keys="$(key_count)"
  echo "keys:      $keys (all dbs)"
  echo "used_memory_human: $(cli INFO memory | tr -d '\r' | sed -n 's/^used_memory_human://p')"
  if [ "$rc" -eq 0 ]; then echo "config:    matches D7 + P1"; else echo "config:    DRIFTED"; fi
  return "$rc"
}

cmd_down() {
  preflight
  local force=0 st n p
  [ "${1:-}" = "--force" ] && force=1
  st="$(state)"
  if [ "$st" = absent ]; then log "$NAME not present — no-op"; return 0; fi
  own_container                       # whatever the state: never stop/rm what we did not create
  if [ "$st" = running ] && [ "$force" -eq 0 ]; then
    # Count the keys of the instance being removed: its own published port, not $PORT.
    p="$(container_port)" || die "$NAME is not published on exactly one $HOST port, so its keys cannot be counted; re-run with --force to discard"
    CLI_PORT="$p"
    n="$(key_count)"
    if [ "$n" != 0 ]; then
      die "$NAME holds $n key(s) — another step's reservation may own them (§5.1). Re-run with --force to discard."
    fi
  fi
  if [ "$st" = running ]; then docker stop "$NAME" >/dev/null </dev/null; fi
  docker rm "$NAME" >/dev/null </dev/null
  log "$NAME stopped and removed (password file kept: $PASS_FILE)"
}

cmd_reset_p0() {
  [ "${1:-}" = "--yes" ] || die "reset-p0 empties $NAME (FLUSHALL). It is approval-gated; re-run with --yes once approved."
  [ "$NAME" = "$SHARED_NAME" ] || die "reset-p0 refuses: container is '$NAME', not $SHARED_NAME"
  [ "$PORT" = "$SHARED_PORT" ] || die "reset-p0 refuses: port is $PORT, not $SHARED_PORT"
  preflight
  [ "$(state)" = running ] || die "$NAME is not running"
  own_container
  [ "$(docker port "$NAME" 6379/tcp </dev/null 2>/dev/null || true)" = "$HOST:$SHARED_PORT" ] \
    || die "reset-p0 refuses: $NAME is not published on exactly $HOST:$SHARED_PORT"
  CLI_PORT="$SHARED_PORT"
  [ "$(cli PING 2>/dev/null || true)" = PONG ] || die "reset-p0: authenticated PING failed (wrong password file?); nothing flushed"
  local before after reply
  before="$(key_count)"
  log "FLUSHALL on $NAME ($HOST:$SHARED_PORT): $before key(s) before"
  reply="$(cli FLUSHALL)"
  [ "$reply" = OK ] || die "reset-p0: FLUSHALL replied '$reply'"
  after="$(key_count)"
  [ "$after" = 0 ] || die "reset-p0: $after key(s) remain after FLUSHALL — NOT done"
  log "done: 0 key(s) after (keyspace only; ACL users and GRAPH.CONFIG persist — for a full clean start use: down --force, then up)"
}

if [ "$THROWAWAY" -eq 1 ]; then
  case "${1:-up}" in
    up)     cmd_tw_up ;;
    status) cmd_tw_need_port status; cmd_status ;;
    down)   cmd_tw_down ;;
    list)   cmd_tw_list ;;
    reset-p0) die "reset-p0 is refused in --throwaway mode (use --throwaway down; a throwaway is disposable)" ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
  exit 0
fi

case "${1:-}" in
  up)       cmd_up ;;
  status)   cmd_status ;;
  down)     shift; cmd_down "$@" ;;
  reset-p0) shift; cmd_reset_p0 "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
