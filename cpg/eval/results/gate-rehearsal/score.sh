#!/usr/bin/env bash
# score.sh <tag> <lists> <key> [extra gate.py score args]   real `gate.py score` on the scratch custodian (env.sh PORT selects the store)
. "$(dirname "$0")/env.sh"
tag=$1; lists=$2; key=$3; shift 3
t0=$(date +%s.%N)
( cd $REPO/cpg/eval/tools && HOME=$HOME_R $PY gate.py score --gate-id $GATE_ID --lists $lists --run $CUST/run-$key.json --custodian $CUST --host 127.0.0.1 --port $PORT --system $SYSTEM "$@" ) > $RH/log/$tag.out 2> $RH/log/$tag.err
echo "exit=$? wall_s=$(echo "$(date +%s.%N) - $t0" | bc)" | tee $RH/log/$tag.rc
