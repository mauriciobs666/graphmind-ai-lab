#!/usr/bin/env bash
# K-098 r4 (U12 gate F2): is the subquery's DISTINCT load-bearing once `e` is referenced? Throwaway k098-gdba-r4d on 16488.
set -uo pipefail
docker run -d --rm --name k098-gdba-r4d -p 127.0.0.1:16488:6379 falkordb/falkordb:v4.18.11 >/dev/null
R="redis-cli -p 16488"; for _ in $(seq 30); do [ "$($R PING 2>/dev/null)" = PONG ] && break; sleep 1; done
# seed S and R share two entities, so (S)->(e)<-(rel) has two paths to each of S and R
$R GRAPH.QUERY g "CREATE (s:Message {msgId:'S'}), (r:Message {msgId:'R'}), (e1:Entity {entityId:'e1'}), (e2:Entity {entityId:'e2'}), (s)-[:MENTIONS]->(e1), (s)-[:MENTIONS]->(e2), (r)-[:MENTIONS]->(e1), (r)-[:MENTIONS]->(e2)" | sed -n 1p
for label in "e unreferenced|" "e filtered|WHERE e.entityId IS NOT NULL "; do
  name=${label%%|*}; w=${label#*|}
  for d in "DISTINCT " ""; do
    q="MATCH (seed:Message {msgId:'S'}) CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) ${w}WITH ${d}related LIMIT 5 RETURN collect(related.msgId) AS ex } RETURN ex"
    echo "$name, ${d:-no DISTINCT}: $($R GRAPH.RO_QUERY g "$q" | sed -n 2,5p | paste -sd' ')"
    $R GRAPH.EXPLAIN g "$q" | grep -E 'Traverse' | sed 's/^ */    /'
  done
done
docker stop k098-gdba-r4d >/dev/null
