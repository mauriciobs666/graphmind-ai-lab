#!/usr/bin/env bash
# rg.sh <tag> <gate_run args...>   one real run through gr.py; logs stdout/stderr/exit/wall to $RH/log/<tag>.*
. "$(dirname "$0")/env.sh"
tag=$1; shift
# W3 (defect D3): the sandbox writer appends to the DEFAULT audit dir while the gate's refresh verifies against the private gate-audit-<key>; point the writer at the
# private dir so the two agree. Without W3 every python/JS unit is not-mapped (bundle:audit-missing). NOWA3=1 reproduces the defect.
if [ -z "$NOWA3" ] && [ "$1" = run ] && [ "$2" = --gate-id ]; then
  LISTS=$(printf '%s\n' "$@" | awk 'p{print; exit} $0=="--lists"{p=1}')
  KEY=$(cd $REPO/cpg/eval/tools && HOME=$HOME_R $PY /home/mauricio/prg/graphmind-ai-lab/cpg/eval/results/gate-rehearsal/key.py "$LISTS")
  export CCG_AUDIT_DIR=$CUST/gate-audit-$KEY
fi
export TIMING_FILE=$RH/log/$tag.timing
t0=$(date +%s.%N)
( cd $REPO/cpg/eval/tools && HOME=$HOME_R $PY /home/mauricio/prg/graphmind-ai-lab/cpg/eval/results/gate-rehearsal/gr.py "$@" ) > $RH/log/$tag.out 2> $RH/log/$tag.err
rc=$?
echo "exit=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc)" | tee $RH/log/$tag.rc
