#!/usr/bin/env bash
# K-098: per-config throwaway container from rdb/dump.rdb (own copy per run), measure, remove. Never touches other containers.
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd); PY=/home/mauricio/prg/graphmind-ai-lab/falkor-chat/server/.venv/bin/python
TC=$1; OMP=$2; NAME=k098-gdba-tc$TC-omp$OMP; PORT=16480
D=$(mktemp -d "$W/rdbrun.XXXX"); cp "$W/rdb/dump.rdb" "$D/dump.rdb"; chmod -R a+rwx "$D"
docker run -d --rm --name $NAME -p 127.0.0.1:$PORT:6379 --memory 2g -e BROWSER=0 -v "$D":/var/lib/falkordb/data \
  -e FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000 THREAD_COUNT $TC OMP_THREAD_COUNT $OMP" falkordb/falkordb:v4.18.11 >/dev/null
until [ "$(redis-cli -p $PORT PING 2>/dev/null)" = PONG ] && redis-cli -p $PORT GRAPH.LIST 2>/dev/null | grep -q ws:synth; do sleep 1; done
echo "## THREAD_COUNT=$(redis-cli -p $PORT GRAPH.CONFIG GET THREAD_COUNT | tail -1) OMP=$(redis-cli -p $PORT GRAPH.CONFIG GET OMP_THREAD_COUNT | tail -1)"
$PY -I "$W/loadgen.py" $PORT 2 5 >/dev/null   # warm caches
for C in 1 8 16 32; do echo "c=$C $($PY -I "$W/loadgen.py" $PORT $C 15)"; done
# single heavy query latency (deep walk) and the instance's own threads
for i in 1 2 3; do redis-cli -p $PORT GRAPH.RO_QUERY ws:deep "MATCH (x:Message {msgId:'d-99999'}) MATCH (t:Thread)-[:HEAD|NEXT*0..]->(x) RETURN t.threadId" timeout 30000 | grep 'execution time'; done
echo "os_threads=$(docker exec $NAME sh -c 'ls /proc/1/task | wc -l')"
docker stop $NAME >/dev/null; rm -rf "$D"
