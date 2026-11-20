#!/usr/bin/env bash
# K-098 r3: re-verify the analyst's denyoom finding — above maxmemory with noeviction, which GRAPH.* commands still run?
set -uo pipefail
docker run -d --rm --name k098-gdba-r3m -p 127.0.0.1:16487:6379 falkordb/falkordb:v4.18.11 >/dev/null
R="redis-cli -p 16487"; for _ in $(seq 30); do [ "$($R PING 2>/dev/null)" = PONG ] && break; sleep 1; done
$R GRAPH.QUERY g "UNWIND range(1,200000) AS i CREATE (:N {id:i, s:'xxxxxxxxxxxxxxxxxxxxxxxx'})" | sed -n 1p
echo "used_memory: $($R INFO memory | grep '^used_memory:' | tr -d '\r')"
$R CONFIG SET maxmemory-policy noeviction; $R CONFIG SET maxmemory 1mb
echo "GRAPH.QUERY read : $($R GRAPH.QUERY g 'MATCH (n) RETURN count(n)' 2>&1 | head -1)"
echo "GRAPH.QUERY write: $($R GRAPH.QUERY g 'CREATE (:N)' 2>&1 | head -1)"
echo "GRAPH.RO_QUERY   : $($R GRAPH.RO_QUERY g 'MATCH (n) RETURN count(n)' 2>&1 | sed -n 2p)"
echo "GRAPH.EXPLAIN    : $($R GRAPH.EXPLAIN g 'MATCH (n) RETURN n' 2>&1 | head -1)"
echo "GRAPH.DELETE     : $($R GRAPH.DELETE g 2>&1 | head -1)"
docker stop k098-gdba-r3m >/dev/null; docker ps -a --format '{{.Names}}' | grep -c k098-gdba-r3m
