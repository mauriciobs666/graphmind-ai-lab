#!/usr/bin/env bash
# K-098 r3 (P2-M1): rehearse note §5 Step 4 incl. the "step 7 differs" rollback branch, with a planted
# wrong-data start. Own container k098-gdba-r3 on 16486, scratch bind dirs under this folder playing
# the live volume and the backup dir. Never falkordb-dev, never ws:test.
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$W/../../../../.." && pwd)
S="$W/../graph-dba-r2/proto/scripts/start_falkordb.sh"           # the edited script (prototype)
IMG=falkordb/falkordb:v4.18.11; C=k098-gdba-r3
LIVE=$(mktemp -d "$W/r3_live.XXXX"); B=$(mktemp -d "$W/r3_backup.XXXX"); chmod a+rwx "$LIVE" "$B"
export FALKORDB_PORT=16486 FALKORDB_WEB_PORT=13086 FALKORDB_CONTAINER_NAME=$C FALKORDB_VOLUME="$LIVE"
R="redis-cli -p 16486"; OLD="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"
wait_up() { for _ in $(seq 60); do [ "$($R PING 2>/dev/null)" = PONG ] && return 0; sleep 1; done; return 1; }
wait_gone() { for _ in $(seq 60); do docker ps -a --format '{{.Names}}' | grep -qx "$C" || return 0; sleep 1; done; return 1; }
inv() { $R GRAPH.LIST | sort | while read -r g; do
  n=$($R GRAPH.RO_QUERY "$g" "MATCH (n) RETURN count(n)" timeout 120000 | sed -n 2p)
  e=$($R GRAPH.RO_QUERY "$g" "MATCH ()-[r]->() RETURN count(r)" timeout 120000 | sed -n 2p)
  i=$($R GRAPH.RO_QUERY "$g" "CALL db.indexes() YIELD label RETURN count(label)" timeout 120000 | sed -n 2p)
  echo "$g nodes=$n edges=$e indexes=$i"; done; }
vsha() { docker run --rm -v "$1":/x --entrypoint sha256sum $IMG /x/dump.rdb | cut -c1-16; }
restore() { docker run --rm -v "$LIVE":/d -v "$B":/b --entrypoint cp $IMG -a /b/dump.rdb /d/dump.rdb; }
plant() { $R GRAPH.QUERY ws:a "MATCH (n:N) WHERE n.id > 13000 DETACH DELETE n" | sed -n 1p
          $R GRAPH.QUERY ws:planted "CREATE (:X)" | sed -n 1p
          echo "BGSAVE (stands for the auto-save 'save 60 10000' persisting the wrong state): $($R BGSAVE)"; sleep 3
          echo "dump.rdb now $(vsha "$LIVE") vs backup $(vsha "$B")"; }

echo "## 0. 'live' on the image's own args, three graphs"
FALKORDB_ARGS="$OLD" bash "$S" -d >/dev/null; wait_up
for g in ws:a ws:b cpg_x; do
  $R GRAPH.QUERY "$g" "UNWIND range(1,20000) AS i CREATE (:N {id:i})-[:R]->(:M {id:i})" >/dev/null
  $R GRAPH.QUERY "$g" "CREATE INDEX FOR (n:N) ON (n.id)" >/dev/null; done
echo "## 2. quiesce: CLIENT LIST shows only this shell; then pause writes for steps 3-5"
echo "clients: $($R CLIENT LIST | wc -l)"; echo "pause: $($R CLIENT PAUSE 1800000 WRITE)"
echo "## 3. inventory before"; inv | tee "$W/r3_inventory-before.txt"
echo "## 4. save"; a=$($R LASTSAVE); sleep 1; $R SAVE; b=$($R LASTSAVE); echo "LASTSAVE advanced=$([ "$b" -gt "$a" ] && echo yes || echo NO)"
echo "## 5. stop, log kept"; docker logs -f $C > "$W/r3_stop.log" 2>&1 & LP=$!
docker stop -t 300 $C >/dev/null; sleep 1; kill $LP 2>/dev/null
grep -cE 'DB saved on disk' "$W/r3_stop.log"; grep -E 'ready to exit' "$W/r3_stop.log" | cut -c30-
echo "## 5b. copy dump.rdb aside"
docker run --rm -v "$LIVE":/d -v "$B":/b --entrypoint cp $IMG -a /d/dump.rdb /b/
echo "live $(vsha "$LIVE") backup $(vsha "$B")"
echo "## 6. start, edited script"; bash "$S" -d | grep 'Module args'; wait_up
echo "## 6x. PLANT wrong data in the new instance"; plant
echo "## 7. inventory after: must differ"; sleep 1; inv > "$W/r3_inventory-planted.txt"
diff "$W/r3_inventory-before.txt" "$W/r3_inventory-planted.txt"; echo "diff exit=$? (want 1: differs -> rollback)"
echo "## 8a. rollback: SHUTDOWN NOSAVE, restore copy, start on old args (edited script)"
$R SHUTDOWN NOSAVE 2>&1; wait_gone && echo "container gone"
restore; echo "restored: live $(vsha "$LIVE")"
FALKORDB_ARGS="$OLD" bash "$S" -d >/dev/null; wait_up; sleep 1
echo "args: $($R GRAPH.CONFIG GET MAX_QUEUED_QUERIES | paste -sd" ") $($R GRAPH.CONFIG GET OMP_THREAD_COUNT | paste -sd" ")"
inv > "$W/r3_inventory-rollback-a.txt"; diff "$W/r3_inventory-before.txt" "$W/r3_inventory-rollback-a.txt"; echo "rollback A inventory diff exit=$? (want 0)"
echo "## 8b. second wrong-data start; rollback via docker kill (redis-cli unusable) + PINNED pre-edit script 2769a99c"
plant
docker kill $C >/dev/null; wait_gone && echo "container gone (SIGKILL, no save)"
restore; echo "restored: live $(vsha "$LIVE")"
git -C "$REPO" show 2769a99c:falkor-chat/scripts/start_falkordb.sh > "$W/r3_start_falkordb.2769a99c.sh"
git -C "$REPO" diff --quiet 2769a99c HEAD -- falkor-chat/scripts/start_falkordb.sh && echo "2769a99c == HEAD (pre-edit) for start_falkordb.sh"
# rehearsal-only patch: the pre-edit script hard-codes the live volume; point it at the scratch dir.
sed "s#-v falkordb-data:#-v ${LIVE}:#" "$W/r3_start_falkordb.2769a99c.sh" > "$W/r3_prev_scratch.sh"
diff "$W/r3_start_falkordb.2769a99c.sh" "$W/r3_prev_scratch.sh" | grep -c '^>'; grep -c 'falkordb-data:' "$W/r3_prev_scratch.sh"
bash "$W/r3_prev_scratch.sh" -d >/dev/null; wait_up; sleep 1
echo "args: $($R GRAPH.CONFIG GET MAX_QUEUED_QUERIES | paste -sd" ") $($R GRAPH.CONFIG GET OMP_THREAD_COUNT | paste -sd" ")"
inv > "$W/r3_inventory-rollback-b.txt"; diff "$W/r3_inventory-before.txt" "$W/r3_inventory-rollback-b.txt"; echo "rollback B inventory diff exit=$? (want 0)"
{ cat "$W/r3_inventory-before.txt"; echo "k098_control nodes=1 edges=0 indexes=0"; } | diff - "$W/r3_inventory-rollback-b.txt" >/dev/null; echo "positive control exit=$? (want 1)"
echo "## cleanup"; docker stop -t 60 $C >/dev/null; wait_gone
docker run --rm --entrypoint rm -v "$W":/w $IMG -rf "/w/$(basename "$LIVE")" "/w/$(basename "$B")"
rm -f "$W/r3_prev_scratch.sh"
docker ps -a --format '{{.Names}}' | grep k098-gdba || echo "rehearsal containers removed"
