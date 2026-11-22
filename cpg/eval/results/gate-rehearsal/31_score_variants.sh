#!/usr/bin/env bash
# Item 9 / N3: score variants on the spent F2 (store 16433) and F3 (store 16434) lists.
D=$(dirname "$(readlink -f "$0")"); cd $D
sv() { # tag port list-name repo [extra]
  tag=$1; port=$2; ln=$3; repo=$4; shift 4
  export PORT=$port; . ./env.sh; K=$(cat $RH/log/k-$ln)
  ( cd $repo/cpg/eval/tools && HOME=$HOME_R $PY gate.py score --gate-id $GATE_ID --lists $CUST/lists/l-$ln.jsonl --run $CUST/run-$K.json --custodian $CUST --host 127.0.0.1 --port $port --system $SYSTEM "$@" ) > $RH/log/$tag.out 2> $RH/log/$tag.err
  echo "$tag exit=$? stdout=$(wc -c < $RH/log/$tag.out)B :: $(head -c 200 $RH/log/$tag.err | tr '\n' ' ')"
}
. ./env.sh
sv s-rerun 16433 f2 $REPO;  cmp <(head -c 100000 $RH/log/f2-score.out) <(head -c 100000 $RH/log/s-rerun.out) && echo "  s-rerun output byte-identical to the first score"
sv s-dirty 16433 f2 $S/repo4
# live-id re-read: change the live :Snapshot id on the F3 throwaway (mine), score, restore
docker exec ccg-rh-nopass-16434 redis-cli GRAPH.QUERY ccg_graphmind-ai-lab "MATCH (s:Snapshot) RETURN s.snapshotId" | sed -n 2p > $RH/log/sid-16434
SID=$(cat $RH/log/sid-16434)
docker exec ccg-rh-nopass-16434 redis-cli GRAPH.QUERY ccg_graphmind-ai-lab "MATCH (s:Snapshot) SET s.snapshotId = '00000000000000000000000000000000' RETURN 1" >/dev/null
sv s-liveid 16434 f3 $REPO
docker exec ccg-rh-nopass-16434 redis-cli GRAPH.QUERY ccg_graphmind-ai-lab "MATCH (s:Snapshot) SET s.snapshotId = '$SID' RETURN 1" >/dev/null
sv s-restored 16434 f3 $REPO
# N3: flow decision flipped to TRUE between run and score (repo2 holds a committed TRUE decision; the F2 run was flag false)
sv s-flowflip 16433 f2 $S/repo2 --flow-decision $S/repo2/cpg/eval/decisions/rehearsal-flow-true.json
$PY scan.py $RH/log/s-*.out $RH/log/s-*.err | tail -1
