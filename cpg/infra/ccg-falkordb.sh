#!/usr/bin/env bash
set -euo pipefail
# ccg-falkordb.sh — the dedicated FalkorDB instance for C-1001 phase P0 (store spikes + slice).
#
# Design: docs/plans/connected-code-graph-p0.md §2 D7, §4.1 (S01), §5.1 (reservations).
# Regression test (stubbed docker/redis-cli, never real containers): tests/test-ccg-falkordb.sh
#
#   up        start the instance if it is not running (idempotent; a running instance is a no-op)
#   status    print state, binding, the D7 config, image/module, memory cap, §5.1 exit-hygiene
#             state and key count (read-only; non-zero exit if the instance is down or drifted)
#   down      stop and remove the container (refuses if its keyspace is non-empty unless
#             --force: keys there may belong to another step's reservation, §5.1)
#   reset-p0  FLUSHALL the instance at P0 exit (§5.1). The ONLY destructive verb; refuses unless
#             the container is ccg-falkordb on 127.0.0.1:6390, and requires --yes. Approval-gated.
#
# Invariants:
#  - OWNERSHIP: no `docker stop`/`docker rm` and no Redis write ever reaches a container that does
#    not carry the label ccg-falkordb=1, which only this script's `docker run` sets. That is what
#    makes a CCG_FALKORDB_NAME override safe: pointing it at `falkordb-dev` (unlabelled) is
#    refused by every verb. The shared `falkordb-dev` (6379) is never stopped, restarted,
#    reconfigured or written to by anything here.
#  - The key-count guard and every Redis call go to the port the target container itself
#    publishes (`docker port`), never to a $CCG_FALKORDB_PORT that might name another instance.
#  - Binds 127.0.0.1 only. `falkordb-dev` publishes 0.0.0.0, so do not copy its `-p` form.
#  - The admin password lives in ~/.config/ccg/falkordb-admin.pass (mode 600, outside the repo),
#    created on first `up` if absent, never printed, never on a command line, never in a
#    container env var (`docker inspect` would show it). It reaches redis-server through a
#    generated config file mounted read-only, and reaches redis-cli through REDISCLI_AUTH.
#  - No persistence (save "", appendonly no), so there is no volume: a removed container loses
#    nothing that was not already lost the moment it stopped. `down --force` + `up` is therefore
#    the full clean start (keyspace, ACL users and runtime config all reset).
#
# Env overrides (defaults are D7's; reset-p0 still refuses anything but the defaults, and status
# compares the image against the pinned default, not against the override):
#   CCG_FALKORDB_NAME   container name          (ccg-falkordb)
#   CCG_FALKORDB_PORT   host port on 127.0.0.1  (6390)
#   CCG_FALKORDB_IMAGE  image                   (falkordb/falkordb:v4.18.11 — pinned, K-P0-9)
#   CCG_CONFIG_DIR      password/config dir     (~/.config/ccg)

PINNED_IMAGE="falkordb/falkordb:v4.18.11"
PINNED_MODULE_VER="41811"            # graph module version shipped in PINNED_IMAGE
NAME="${CCG_FALKORDB_NAME:-ccg-falkordb}"
PORT="${CCG_FALKORDB_PORT:-6390}"
IMAGE="${CCG_FALKORDB_IMAGE:-$PINNED_IMAGE}"
CFG_DIR="${CCG_CONFIG_DIR:-$HOME/.config/ccg}"
PASS_FILE="$CFG_DIR/falkordb-admin.pass"
CONF_FILE="$CFG_DIR/$NAME.redis.conf"
HOST=127.0.0.1
LABEL_KEY="ccg-falkordb"
CLI_PORT="$PORT"                     # re-pointed at the container's own published port

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

log() { echo "ccg-falkordb.sh: $*" >&2; }
die() { log "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: ccg-falkordb.sh <up|status|down [--force]|reset-p0 --yes|-h>

  up        start $NAME on $HOST:$PORT if not running (idempotent)
  status    report state + D7 config + exit hygiene; non-zero exit if down or drifted
  down      stop and remove $NAME (refuses a non-empty keyspace without --force)
  reset-p0  FLUSHALL $NAME (P0 exit only; needs --yes; approval-gated)

Full clean start (keyspace, ACL users, runtime config): down --force, then up.
Connect:  REDISCLI_AUTH="\$(cat $PASS_FILE)" redis-cli -h $HOST -p $PORT
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

label_of() {
  docker inspect "$NAME" --format "{{index .Config.Labels \"$LABEL_KEY\"}}" 2>/dev/null </dev/null || true
}

# The ownership gate. Called before every docker stop/rm and every Redis write.
own_container() {
  [ "$(label_of)" = 1 ] || die "refusing: container '$NAME' does not carry the label $LABEL_KEY=1, so this script did not create it (it never stops, removes or writes to a container it does not own)"
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
  local pass
  pass="$(cat "$PASS_FILE")"
  (umask 077 && cat >"$CONF_FILE" <<EOF
# generated by cpg/infra/ccg-falkordb.sh — do not edit, it is rewritten on every \`up\`
requirepass $pass
maxmemory $MAXMEMORY
maxmemory-policy $POLICY
save ""
appendonly no
EOF
  )
  chmod 600 "$CONF_FILE"
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

port_in_use() {
  ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .
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
  ensure_password
  write_conf
  log "starting $NAME ($IMAGE) on $HOST:$PORT, memory cap $MEM_CAP (no swap)"
  # BROWSER=0: skip the bundled Node web console (saves RAM on the 6 GB host; no port 3000).
  # REDIS_ARGS: the image's run.sh puts it first on redis-server's argv, which is where a
  #   config-file path must go. FALKORDB_ARGS is deliberately NOT passed, so the image's
  #   module defaults stay as they are on falkordb-dev (bare `-e VAR` would delete them).
  # --pull=never: the pinned image is already local; a miss should say so, not pull.
  docker run -d \
    --name "$NAME" \
    --label "$LABEL_KEY=1" \
    --pull=never \
    --memory "$MEM_CAP" --memory-swap "$MEM_CAP" \
    -p "$HOST:$PORT:6379" \
    -e BROWSER=0 \
    -e REDIS_ARGS=/etc/ccg/redis.conf \
    -v "$CONF_FILE:/etc/ccg/redis.conf:ro" \
    "$IMAGE" >/dev/null </dev/null
  CLI_PORT="$PORT"
  wait_ready
  log "$NAME up"
}

cmd_status() {
  preflight
  local st rc=0 lab img ports nports p
  drift() { echo "DRIFT: $*"; rc=1; }
  st="$(state)"
  echo "container: $NAME"
  echo "state:     $st"
  if [ "$st" != running ]; then
    echo "(not running — start with: $0 up)"
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

  local mm pol save aof modver tmax acl keys
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
  # §5.1 exit hygiene: exclusive steps restore TIMEOUT_MAX to 0 and delete their ACL users.
  tmax="$(cli GRAPH.CONFIG GET TIMEOUT_MAX | sed -n 2p)"
  acl="$(cli ACL USERS | paste -sd' ' -)"
  echo "TIMEOUT_MAX: $tmax (default 0)"
  echo "ACL users:   $acl (expected: default)"
  [ "$tmax" = 0 ]       || drift "GRAPH.CONFIG TIMEOUT_MAX is $tmax, not 0 (§5.1 exit hygiene)"
  [ "$acl" = default ]  || drift "ACL users besides default: $acl (§5.1 exit hygiene)"
  keys="$(key_count)"
  echo "keys:      $keys (all dbs)"
  echo "used_memory_human: $(cli INFO memory | tr -d '\r' | sed -n 's/^used_memory_human://p')"
  if [ "$rc" -eq 0 ]; then echo "config:    matches D7"; else echo "config:    DRIFTED from D7"; fi
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
  [ "$NAME" = ccg-falkordb ] || die "reset-p0 refuses: container is '$NAME', not ccg-falkordb"
  [ "$PORT" = 6390 ]         || die "reset-p0 refuses: port is $PORT, not 6390"
  preflight
  [ "$(state)" = running ] || die "$NAME is not running"
  own_container
  [ "$(docker port "$NAME" 6379/tcp </dev/null 2>/dev/null || true)" = "$HOST:6390" ] \
    || die "reset-p0 refuses: $NAME is not published on exactly $HOST:6390"
  CLI_PORT=6390
  [ "$(cli PING 2>/dev/null || true)" = PONG ] || die "reset-p0: authenticated PING failed (wrong password file?); nothing flushed"
  local before after reply
  before="$(key_count)"
  log "FLUSHALL on $NAME ($HOST:6390): $before key(s) before"
  reply="$(cli FLUSHALL)"
  [ "$reply" = OK ] || die "reset-p0: FLUSHALL replied '$reply'"
  after="$(key_count)"
  [ "$after" = 0 ] || die "reset-p0: $after key(s) remain after FLUSHALL — NOT done"
  log "done: 0 key(s) after (keyspace only; ACL users and GRAPH.CONFIG persist — for a full clean start use: down --force, then up)"
}

case "${1:-}" in
  up)       cmd_up ;;
  status)   cmd_status ;;
  down)     shift; cmd_down "$@" ;;
  reset-p0) shift; cmd_reset_p0 "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
