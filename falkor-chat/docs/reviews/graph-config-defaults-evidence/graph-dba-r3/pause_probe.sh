#!/usr/bin/env bash
# K-098 r3: does CLIENT PAUSE ... WRITE hold GRAPH.QUERY writes while RO_QUERY, SAVE and a SIGTERM
# shutdown-save still work? Throwaway container k098-gdba-r3p on 16485 only, no volume.
set -uo pipefail
docker run -d --rm --name k098-gdba-r3p -p 127.0.0.1:16485:6379 falkordb/falkordb:v4.18.11 >/dev/null
R="redis-cli -p 16485"; for _ in $(seq 30); do [ "$($R PING 2>/dev/null)" = PONG ] && break; sleep 1; done
$R GRAPH.QUERY g "UNWIND range(1,100) AS i CREATE (:N {id:i})" | tail -n +1 | head -2
echo "pause: $($R CLIENT PAUSE 20000 WRITE)"
s=$(date +%s.%N); timeout 4 $R GRAPH.QUERY g "CREATE (:N {id:-1})"; echo "write via GRAPH.QUERY exit=$? after $(echo "$(date +%s.%N)-$s"|bc)s (124 = held by pause)"
echo "read via GRAPH.RO_QUERY: $($R GRAPH.RO_QUERY g 'MATCH (n) RETURN count(n)' | sed -n 2p)"
s=$(date +%s.%N); echo "read via GRAPH.QUERY (no write clause): $(timeout 4 $R GRAPH.QUERY g 'MATCH (n) RETURN count(n)' | sed -n 2p) exit=${PIPESTATUS[0]} $(echo "$(date +%s.%N)-$s"|bc)s"
echo "SAVE while paused: $(timeout 10 $R SAVE)"
echo "count after pause window (write must not have landed yet): $($R GRAPH.RO_QUERY g 'MATCH (n) RETURN count(n)' | sed -n 2p)"
docker logs -f k098-gdba-r3p > pause_stop.log 2>&1 & LP=$!
s=$(date +%s); docker stop -t 60 k098-gdba-r3p >/dev/null; echo "docker stop while paused took $(( $(date +%s)-s ))s"; sleep 1; kill $LP 2>/dev/null
grep -E 'DB saved on disk|ready to exit|paus' pause_stop.log
docker ps -a --format '{{.Names}}' | grep -c k098-gdba-r3p
