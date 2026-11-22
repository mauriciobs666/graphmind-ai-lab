#!/usr/bin/env bash
# usage: 22_continue2.sh <f4|f5|f6|f7|f8|f9>   the steps after the first (fresh) run of a scenario: state, resume(s), score. Logs under $RH/log.
D=$(dirname "$(readlink -f "$0")"); cd $D
args() { . ./env.sh; echo run --gate-id $GATE_ID --custodian $CUST --repo $REPO --system $SYSTEM --host 127.0.0.1 --port $PORT --lists $CUST/lists/$1.jsonl; }
score() { # name port [repo]
  PORT=$2; . ./env.sh; ./key.sh $CUST/lists/l-$1.jsonl > $RH/log/k-$1
  [ -n "$3" ] && { REPO=$3; . ./env.sh; }
  ./score.sh $1-score $CUST/lists/l-$1.jsonl $(cat $RH/log/k-$1)
  echo "  score stderr: $(head -c 200 $RH/log/$1-score.err | tr '\n' ' ')"
  $PY - $1 <<'P'
import json, os, sys
try:
    d = json.load(open(os.environ["RH"] + f"/log/{sys.argv[1]}-score.out"))
except Exception as e:
    print("  (no score document)"); sys.exit()
print("  verdict", d["verdict"], "H", d["H"], "blockers", d["blockers"], "harnessErrors", d["harnessErrors"], "engineErrors", json.dumps(d["engineErrors"]), "stops", d["stops"], "voided", d["voided"], "flow", d["flow"], "label", d["label"])
P
  $PY scan.py $RH/log/$1-score.out $RH/log/$1-score.err | tail -1
}
scan_all() { $PY scan.py $RH/log/$1*.out $RH/log/$1*.err | tail -1; }
case "$1" in
f4)
  export PORT=16435; . ./env.sh; L=$CUST/lists/l-f4.jsonl
  echo "[after SIGKILL before the final marker write]"; ./state.sh $L
  ./key.sh $L > $RH/log/k-f4; ./score.sh f4-score-early $L $(cat $RH/log/k-f4); echo "  early score stderr: $(cat $RH/log/f4-score-early.err | tr '\n' ' ')"
  ./rg.sh f4b $(args l-f4); echo "  re-run without --resume: $(cat $RH/log/f4b.err | tr '\n' ' ')"
  ./rg.sh f4c $(args l-f4) --resume; echo "[after resume]"; ./state.sh $L; cat $RH/log/f4c.out; echo "  stderr: $(cat $RH/log/f4c.err | tr '\n' ' ')"; cat $RH/log/f4c.timing
  scan_all f4; score f4 16435 ;;
f5)
  export PORT=16436; . ./env.sh; L=$CUST/lists/l-f5.jsonl
  echo "[after the first stop: ValueError engine-error at ask 3, TimeoutError at ask 8]"; ./state.sh $L; echo "  stderr: $(cat $RH/log/f5a.err | tr '\n' ' ')"
  FAULT=1:TimeoutError ./rg.sh f5b $(args l-f5) --resume; echo "[after stop 2 (same anchor)]"; ./state.sh $L; echo "  stderr: $(cat $RH/log/f5b.err | tr '\n' ' ')"
  FAULT=1:TimeoutError ./rg.sh f5c $(args l-f5) --resume; echo "[after stop 3 (converted)]"; ./state.sh $L; cat $RH/log/f5c.out; echo "  stderr: $(cat $RH/log/f5c.err | tr '\n' ' ')"
  scan_all f5; score f5 16436 ;;
f6)
  export PORT=16437; . ./env.sh; L=$CUST/lists/l-f6.jsonl
  echo "[stale engine stamp at ask 3]"; ./state.sh $L; cat $RH/log/f6a.out; echo "  stderr: $(cat $RH/log/f6a.err | tr '\n' ' ')"
  scan_all f6; score f6 16437 ;;
f7)
  export PORT=16438; . ./env.sh; L=$CUST/lists/l-f7.jsonl
  echo "[BadRequest at ask 4, ENOSPC at ask 9]"; ./state.sh $L; echo "  stderr: $(cat $RH/log/f7a.err | tr '\n' ' ')"
  ./rg.sh f7b $(args l-f7) --resume; echo "[after resume]"; ./state.sh $L; cat $RH/log/f7b.out; echo "  stderr: $(cat $RH/log/f7b.err | tr '\n' ' ')"
  scan_all f7; score f7 16438 ;;
f8)
  export PORT=16439; . ./env.sh; L=$CUST/lists/l-f8.jsonl
  echo "[run-wide deterministic fault at answer-line 7]"; ./state.sh $L; echo "  stderr: $(cat $RH/log/f8a.err | tr '\n' ' ')"
  ./rg.sh f8b $(args l-f8) --resume; echo "  resume of failed:deterministic: exit $(cat $RH/log/f8b.rc | cut -d' ' -f1) :: $(cat $RH/log/f8b.err | tr '\n' ' ')"
  ./key.sh $L > $RH/log/k-f8; ./score.sh f8-score $L $(cat $RH/log/k-f8); echo "  score: $(cat $RH/log/f8-score.err | tr '\n' ' ')"
  scan_all f8 ;;
f9)
  . ./env.sh; export PORT=16437 REPO=$S/repo2; . ./env.sh; L=$CUST/lists/l-fl.jsonl
  echo "[flow flag true, --allow-after-adaptation]"; ./state.sh $L; cat $RH/log/f9a.out; echo "  stderr: $(cat $RH/log/f9a.err | tr '\n' ' ')"; cat $RH/log/f9a.timing
  ./key.sh $L > $RH/log/k-fl; ./score.sh fl-score $L $(cat $RH/log/k-fl) --flow-decision $S/repo2/cpg/eval/decisions/rehearsal-flow-true.json
  echo "  score stderr: $(head -c 200 $RH/log/fl-score.err | tr '\n' ' ')"
  $PY - <<'P'
import json, os
d = json.load(open(os.environ["RH"] + "/log/fl-score.out"))
print("  verdict", d["verdict"], "H", d["H"], "label", d["label"], "flow", d["flow"], "afterAdaptationAllowed", d["preflight"]["afterAdaptationAllowed"], "harnessPin", d["preflight"]["harnessPin"]["afterAdaptation"], d["preflight"]["harnessPin"]["differ"])
print("  recall items", {k: (v["items"]["k"], v["items"]["n"]) for k, v in d["recall"]["byStratum"].items()})
P
  scan_all f9; scan_all fl ;;
esac
