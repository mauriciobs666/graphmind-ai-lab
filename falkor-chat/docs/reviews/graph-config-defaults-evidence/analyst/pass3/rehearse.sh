#!/usr/bin/env bash
# analyst Pass 3: independent rehearsal of note §5 Step 4 (91c53a2d), commands as written, with a NAMED
# docker volume standing in for falkordb-data. Own container analyst-k098-p3 on 16597. Never falkordb-dev.
set -uo pipefail
REPO=/home/mauricio/prg/graphmind-ai-lab
ES=$REPO/falkor-chat/docs/reviews/graph-config-defaults-evidence/graph-dba-r2/proto/scripts/start_falkordb.sh
W=$(cd "$(dirname "$0")" && pwd); IMG=falkordb/falkordb:v4.18.11; C=analyst-k098-p3; VOL=analyst-k098-live
export FALKORDB_PORT=16597 FALKORDB_WEB_PORT=13097 FALKORDB_CONTAINER_NAME=$C FALKORDB_VOLUME=$VOL
R="redis-cli -p 16597"; OLD="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"
up() { for _ in $(seq 90); do [ "$($R PING 2>/dev/null)" = PONG ] && return 0; sleep 1; done; echo "NOT UP"; return 1; }
gone() { for _ in $(seq 60); do docker ps -a --format '{{.Names}}' | grep -qx "$C" || return 0; sleep 1; done; return 1; }
inv() { $R GRAPH.LIST | sort | while read -r g; do
  n=$($R GRAPH.RO_QUERY "$g" "MATCH (n) RETURN count(n)" timeout 120000 | sed -n 2p)
  e=$($R GRAPH.RO_QUERY "$g" "MATCH ()-[r]->() RETURN count(r)" timeout 120000 | sed -n 2p)
  i=$($R GRAPH.RO_QUERY "$g" "CALL db.indexes() YIELD label RETURN count(label)" timeout 120000 | sed -n 2p)
  echo "$g nodes=$n edges=$e indexes=$i"; done; }
docker volume create $VOL >/dev/null
echo "## seed 'live' (old args, BROWSER on as in falkordb-dev)"; FALKORDB_ARGS="$OLD" bash "$ES" -d >/dev/null; up
for g in ws:a cpg_y; do $R GRAPH.QUERY $g "UNWIND range(1,30000) AS i CREATE (:N {id:i})-[:R]->(:M {id:i})" >/dev/null; $R GRAPH.QUERY $g "CREATE INDEX FOR (n:N) ON (n.id)" >/dev/null; done
sleep 8; echo "## step 2 CLIENT LIST lines with browser up, no other client: $($R CLIENT LIST | wc -l)"; $R CLIENT LIST | sed 's/ addr=.*cmd=/ cmd=/' | cut -c1-80
echo "pause: $($R CLIENT PAUSE 1800000 WRITE)"
echo "## step 3"; inv > $W/inv-before.txt; cat $W/inv-before.txt
echo "## step 4"; a=$($R LASTSAVE); sleep 1; $R SAVE; b=$($R LASTSAVE); echo "LASTSAVE advanced: $([ "$b" -gt "$a" ] && echo yes || echo NO)"
echo "## step 5"; docker logs -f $C > $W/stop.log 2>&1 & LP=$!; docker stop -t 300 $C >/dev/null; sleep 1; kill $LP 2>/dev/null
tail -2 $W/stop.log | cut -c30-
B=$(mktemp -d $W/backup.XXXX) && docker run --rm -v $VOL:/d -v "$B":/b --entrypoint cp $IMG -a /d/dump.rdb /b/; echo "backup: $(ls -l $B | tail -1 | awk '{print $3, $5}')"
gone; echo "## step 6 (edited script, new args)"; bash "$ES" -d | grep 'Module args'; up
echo "## plant wrong data + auto-save stand-in"; $R GRAPH.QUERY ws:a "MATCH (n:N) WHERE n.id > 20000 DETACH DELETE n" | sed -n 1p; $R BGSAVE; sleep 4
echo "## step 7"; inv > $W/inv-after.txt; diff -q $W/inv-before.txt $W/inv-after.txt >/dev/null; echo "diff exit=$? (want 1)"
echo "## step 8 branch A, verbatim lines"
$R SHUTDOWN NOSAVE || docker kill $C; gone
docker run --rm -v $VOL:/d -v "$B":/b --entrypoint cp $IMG -a /b/dump.rdb /d/dump.rdb
FALKORDB_ARGS="$OLD" bash "$ES" -d >/dev/null; up; sleep 2
inv > $W/inv-rbA.txt; diff $W/inv-before.txt $W/inv-rbA.txt; echo "branch A diff exit=$? (want 0); MQQ=$($R GRAPH.CONFIG GET MAX_QUEUED_QUERIES | tail -1)"
echo "## step 8 branch B: unresponsive instance (SIGSTOP the server so redis-cli cannot answer) -> fallback docker kill; pinned 2769a99c"
$R GRAPH.QUERY ws:a "MATCH (n:N) WHERE n.id > 10000 DETACH DELETE n" | sed -n 1p; $R BGSAVE; sleep 4
docker exec $C sh -c 'kill -STOP 1' 2>/dev/null || docker pause $C >/dev/null
( timeout 15 $R SHUTDOWN NOSAVE ) || docker kill $C; docker unpause $C >/dev/null 2>&1; gone; echo "gone after fallback"
docker run --rm -v $VOL:/d -v "$B":/b --entrypoint cp $IMG -a /b/dump.rdb /d/dump.rdb
git -C $REPO show 2769a99c:falkor-chat/scripts/start_falkordb.sh | sed "s#-v falkordb-data:#-v $VOL:#" > $W/prev_scratch.sh
echo "pinned script volume refs left to live volume: $(grep -c 'falkordb-data:' $W/prev_scratch.sh)"
bash $W/prev_scratch.sh -d >/dev/null; up; sleep 2
inv > $W/inv-rbB.txt; diff $W/inv-before.txt $W/inv-rbB.txt; echo "branch B diff exit=$? (want 0)"
{ cat $W/inv-before.txt; echo "k098_control nodes=1 edges=0 indexes=0"; } | diff - $W/inv-rbB.txt >/dev/null; echo "positive control exit=$? (want 1)"
docker rm -f $C >/dev/null; gone; docker volume rm $VOL >/dev/null; docker run --rm -v "$W":/w --entrypoint rm $IMG -rf "/w/$(basename $B)"; rm -f $W/prev_scratch.sh
docker ps -a --format '{{.Names}}' | grep -c analyst; docker volume ls -q | grep -c analyst
