#!/usr/bin/env bash
# usage: 21_continue.sh <f2|f3|...>   the steps after the first (fresh) run of a scenario: state, resume(s), score. Every step logs under $RH/log.
D=$(dirname "$(readlink -f "$0")"); cd $D
args() { . ./env.sh; echo run --gate-id $GATE_ID --custodian $CUST --repo $REPO --system $SYSTEM --host 127.0.0.1 --port $PORT --dev-probe $DEV_PROBE --lists $CUST/lists/$1.jsonl; }
sc() { # name port
  PORT=$2; . ./env.sh; ./key.sh $CUST/lists/l-$1.jsonl > $RH/log/k-$1; ./score.sh $1-score $CUST/lists/l-$1.jsonl $(cat $RH/log/k-$1)
  echo "score: $(head -c 200 $RH/log/$1-score.err | tr '\n' ' ')"; $PY scan.py $RH/log/$1-score.out $RH/log/$1-score.err | tail -1
}
case "$1" in
f2)
  export PORT=16433; . ./env.sh; L=$CUST/lists/l-f2.jsonl
  echo "[after SIGKILL at ask 10]"; ./state.sh $L
  KILL_AT=ask:5 KILL_SIG=TERM ./rg.sh f2b $(args l-f2) --resume; echo "[after resume + SIGTERM at ask 5]"; ./state.sh $L; echo "stderr: $(cat $RH/log/f2b.err | tr '\n' ' ')"
  ./rg.sh f2c $(args l-f2) --resume; echo "[after second resume]"; ./state.sh $L; cat $RH/log/f2c.out
  $PY scan.py $RH/log/f2a.out $RH/log/f2a.err $RH/log/f2b.out $RH/log/f2b.err $RH/log/f2c.out $RH/log/f2c.err | tail -1
  sc f2 16433 ;;
f3)
  export PORT=16434; . ./env.sh; L=$CUST/lists/l-f3.jsonl
  echo "[after SIGKILL before the run.json write]"; ./state.sh $L
  ./rg.sh f3b $(args l-f3) --resume; echo "[after resume]"; ./state.sh $L; cat $RH/log/f3b.out; echo "stderr: $(cat $RH/log/f3b.err | tr '\n' ' ')"
  $PY scan.py $RH/log/f3a.out $RH/log/f3a.err $RH/log/f3b.out $RH/log/f3b.err | tail -1
  sc f3 16434 ;;
esac
