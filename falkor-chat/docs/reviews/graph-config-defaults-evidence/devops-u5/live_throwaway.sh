#!/usr/bin/env bash
# U5 live check: start THROWAWAY FalkorDB containers (same pinned image) through the edited
# scripts/start_falkordb.sh, on scratch host dirs and scratch ports, read every value back,
# and show a positive control (image defaults) and a bad-value start. Never touches falkordb-dev.
set -uo pipefail
cd "$(dirname "$0")/../../../.."   # falkor-chat/
KEYS="THREAD_COUNT OMP_THREAD_COUNT CACHE_SIZE MAX_QUEUED_QUERIES QUERY_MEM_CAPACITY ASYNC_DELETE TIMEOUT RESULTSET_SIZE"
wait_ping() { for _ in $(seq 1 40); do [ "$(redis-cli -p "$1" PING 2>/dev/null)" = PONG ] && return 0; sleep 0.5; done; return 1; }
readback() { for k in $KEYS; do printf '%s=%s\n' "$k" "$(redis-cli -p "$1" GRAPH.CONFIG GET "$k" | sed -n 2p)"; done; }
run() { # name port webport [env assignments...]
  local name=$1 port=$2 web=$3; shift 3
  local d; d=$(mktemp -d); chmod a+rwx "$d"
  echo "### $name: env $* FALKORDB_VOLUME=<scratch dir> FALKORDB_PORT=$port ./scripts/start_falkordb.sh -d"
  env "$@" FALKORDB_VOLUME="$d" FALKORDB_PORT="$port" FALKORDB_WEB_PORT="$web" \
      FALKORDB_CONTAINER_NAME="$name" ./scripts/start_falkordb.sh -d >"$d.out" 2>&1
  echo "start_falkordb.sh exit=$?"; grep -E 'Module args|Data volume' "$d.out" | sed "s#$d#<scratch dir>#"
  if wait_ping "$port"; then
    echo "-- MODULE LIST args:"; redis-cli -p "$port" MODULE LIST | sed -n '/^args$/,$p' | tr '\n' ' '; echo
    echo "-- GRAPH.CONFIG GET:"; readback "$port"
    echo "-- volume mount:"; docker inspect "$name" --format '{{range .Mounts}}{{.Type}} -> {{.Destination}}{{end}}'
    docker stop -t 5 "$name" >/dev/null && echo "stopped (--rm removes it)"
  else
    echo "-- no PONG within 20 s; container state: $(docker inspect -f '{{.State.Status}}' "$name" 2>&1)"
  fi
  rm -rf "$d" "$d.out" 2>/dev/null || docker run --rm -v "$d":/d --entrypoint sh falkordb/falkordb:v4.18.11 -c 'rm -rf /d/*' >/dev/null; rm -rf "$d" "$d.out"
  echo
}
run k098-devops-u5-new  16399 13099
run k098-devops-u5-tc4  16398 13098 FALKORDB_THREAD_COUNT=4
run k098-devops-u5-ctrl 16397 13097 "FALKORDB_ARGS=MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"
BAD="OMP_THREAD_COUNT 1 CACHE_SIZE 25 MAX_QUEUED_QUERIES 320 QUERY_MEM_CAPACITY 67108864 ASYNC_DELETE 1 TIMEOUT 1000 RESULTSET_SIZE 10000"
run k098-devops-u5-bad  16396 13096 "FALKORDB_MODULE_ARGS=$BAD"
echo "### bad value, same args without --rm so the abort log survives"
docker run -d --name k098-devops-u5-badlog -e FALKORDB_ARGS="THREAD_COUNT 8 $BAD" falkordb/falkordb:v4.18.11 >/dev/null
sleep 4
echo "state=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' k098-devops-u5-badlog)"
docker logs k098-devops-u5-badlog 2>&1 | grep -E 'falkordb.so|aborting|Ready to accept'
docker rm k098-devops-u5-badlog >/dev/null && echo "removed k098-devops-u5-badlog"
echo
echo "### leftovers (must be empty):"; docker ps -a --format '{{.Names}}' | grep k098-devops-u5 || echo "(none)"
echo "### falkordb-dev untouched:"; docker ps --filter name=^falkordb-dev$ --format '{{.Names}} {{.Status}}'
