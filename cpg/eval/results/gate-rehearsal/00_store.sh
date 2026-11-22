#!/usr/bin/env bash
# The rehearsal's own throwaway store. usage: 00_store.sh up|down|reset
# "nopass": default user WITHOUT requirepass. The lab's standard throwaway (requirepass) makes the harness's plain FalkorDB(host, port) reader
# fail with AuthenticationError (defect D1); this variant is the rehearsal-setup workaround that lets the rest of the run be exercised.
. "$(dirname "$0")/env.sh"
NAME=ccg-rh-nopass-$PORT
case "$1" in
up)
  docker run -d --name $NAME --label ccg-falkordb.rehearsal=1 --pull=never --memory 2g --memory-swap 2g -p 127.0.0.1:$PORT:6379 -e BROWSER=0 falkordb/falkordb:v4.18.11 >/dev/null
  for i in $(seq 1 40); do docker exec $NAME redis-cli ping 2>/dev/null | grep -q PONG && break; sleep 1; done
  docker exec $NAME redis-cli GRAPH.CONFIG SET TIMEOUT_MAX 10000; docker exec $NAME redis-cli GRAPH.CONFIG SET MAX_INFO_QUERIES 0
  docker exec $NAME redis-cli GRAPH.CONFIG GET TIMEOUT_MAX | tail -1 ;;
down) docker rm -f $NAME >/dev/null && echo removed $NAME ;;
reset) "$0" down; "$0" up; "$(dirname "$0")/02_provision.sh" ;;
esac
