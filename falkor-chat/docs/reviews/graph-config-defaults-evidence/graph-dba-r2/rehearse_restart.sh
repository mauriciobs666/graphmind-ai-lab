#!/usr/bin/env bash
# K-098 r2 (M2): rehearse note §5 Step 4 with the PROTOTYPE script against a scratch data dir that
# plays the live volume. Own container names/ports only (k098-gdba-rh*, 16484/16399); never falkordb-dev.
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd); S="$W/proto/scripts/start_falkordb.sh"
LIVE=$(mktemp -d "$W/rh_live.XXXX"); chmod a+rwx "$LIVE"
export FALKORDB_PORT=16484 FALKORDB_WEB_PORT=13084 FALKORDB_CONTAINER_NAME=k098-gdba-rh FALKORDB_VOLUME="$LIVE"
R="redis-cli -p 16484"
wait_up() { for _ in $(seq 60); do [ "$(redis-cli -p "$1" PING 2>/dev/null)" = PONG ] && return 0; sleep 1; done; return 1; }
inv() { $R GRAPH.LIST | sort | while read -r g; do
  n=$($R GRAPH.RO_QUERY "$g" "MATCH (n) RETURN count(n)" timeout 120000 | sed -n 2p)
  e=$($R GRAPH.RO_QUERY "$g" "MATCH ()-[r]->() RETURN count(r)" timeout 120000 | sed -n 2p)
  i=$($R GRAPH.RO_QUERY "$g" "CALL db.indexes() YIELD label RETURN count(label)" timeout 120000 | sed -n 2p)
  echo "$g nodes=$n edges=$e indexes=$i"; done; }

echo "## 0. 'live' instance on the old image args, seeded with three graphs"
FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000" bash "$S" -d >/dev/null; wait_up 16484
for g in ws:a ws:b cpg_x; do
  $R GRAPH.QUERY "$g" "UNWIND range(1,20000) AS i CREATE (:N {id:i})-[:R]->(:M {id:i})" >/dev/null
  $R GRAPH.QUERY "$g" "CREATE INDEX FOR (n:N) ON (n.id)" >/dev/null; done
echo "db.indexes() columns: $($R GRAPH.RO_QUERY ws:a 'CALL db.indexes() YIELD label, status RETURN label, status' 2>&1 | sed -n '1,4p' | paste -sd' ')"

echo "## 1. dry run of the edited script on a scratch dir (not the 'live' one), spare port"
DRY=$(mktemp -d "$W/rh_dry.XXXX"); chmod a+rwx "$DRY"
FALKORDB_VOLUME="$DRY" FALKORDB_PORT=16399 FALKORDB_WEB_PORT=13099 FALKORDB_CONTAINER_NAME=k098-gdba-rhdry bash "$S" -d | grep 'Module args'
wait_up 16399; redis-cli -p 16399 MODULE LIST | paste -sd' ' | sed 's/.*args //'; redis-cli -p 16399 GRAPH.CONFIG GET MAX_QUEUED_QUERIES | paste -sd' '
docker stop k098-gdba-rhdry >/dev/null

echo "## 3. inventory before"; inv | tee "$W/rh_inventory-before.txt"
echo "## 4. save"; a=$($R LASTSAVE); sleep 1; s=$(date +%s.%N); $R SAVE; echo "SAVE took $(echo "$(date +%s.%N) - $s" | bc) s"; b=$($R LASTSAVE); echo "LASTSAVE $a -> $b advanced=$([ "$b" -gt "$a" ] && echo yes || echo NO)"
echo "## 5. stop with shutdown log captured despite --rm"
docker logs -f k098-gdba-rh > "$W/rh_stop.log" 2>&1 & LP=$!
docker stop -t 300 k098-gdba-rh >/dev/null; sleep 1; kill $LP 2>/dev/null
grep -E 'DB saved on disk|ready to exit|Saving the final' "$W/rh_stop.log"
echo "## 6. start with the edited script (env-file args)"; bash "$S" -d | grep 'Module args'; wait_up 16484
echo "## 7. inventory after + diff + positive control"
sleep 2; inv > "$W/rh_inventory-after.txt"
diff "$W/rh_inventory-before.txt" "$W/rh_inventory-after.txt"; echo "diff exit=$? (want 0)"
{ cat "$W/rh_inventory-before.txt"; echo "k098_control nodes=1 edges=0 indexes=0"; } > "$W/rh_before_plus_control.txt"
diff "$W/rh_before_plus_control.txt" "$W/rh_inventory-after.txt" >/dev/null; echo "control diff exit=$? (want 1)"
$R GRAPH.CONFIG GET '*' | paste - - | grep -E 'THREAD|CACHE|QUEUED|MEM_CAP|ASYNC|^TIMEOUT\b|RESULTSET'
echo "## 8. rollback drill: bad args abort the start; rollback command restores the old args on the same data"
docker stop -t 300 k098-gdba-rh >/dev/null
FALKORDB_ARGS="ASYNC_DELETE 1" bash "$S" -d >/dev/null; sleep 4
echo "bad start running? $(docker ps --format '{{.Names}}' | grep -c '^k098-gdba-rh$')"
FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000" bash "$S" -d >/dev/null; wait_up 16484
echo "after rollback: $($R MODULE LIST | paste -sd' ' | sed 's/.*args //')"; inv | diff "$W/rh_inventory-before.txt" - ; echo "rollback inventory diff exit=$? (want 0)"
docker stop -t 300 k098-gdba-rh >/dev/null
docker run --rm --entrypoint rm -v "$W":/w falkordb/falkordb:v4.18.11 -rf "/w/$(basename "$LIVE")" "/w/$(basename "$DRY")"
docker ps -a --format '{{.Names}}' | grep k098-gdba-rh || echo "rehearsal containers removed"
