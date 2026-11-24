#!/usr/bin/env bash
# usage: 20_scenarios.sh <name> -- launches the first (fresh) run of a scenario on its own store; the resume/score steps are separate commands.
. "$(dirname "$0")/env.sh"
args() { echo run --gate-id $GATE_ID --custodian $CUST --repo $REPO --system $SYSTEM --host 127.0.0.1 --port $PORT --dev-probe $DEV_PROBE --lists $CUST/lists/$1.jsonl; }
cd "$(dirname "$0")"
. ./env.sh
case "$1" in
f2) PORT=16433 KILL_AT=ask:10 KILL_SIG=KILL ./rg.sh f2a $(PORT=16433 args l-f2) ;;
f3) PORT=16434 KILL_AT=runfile ./rg.sh f3a $(PORT=16434 args l-f3) ;;
f4) PORT=16435 KILL_AT=marker-asked ./rg.sh f4a $(PORT=16435 args l-f4) ;;
f5) PORT=16436 FAULT=3:ValueError,8:TimeoutError ./rg.sh f5a $(PORT=16436 args l-f5) ;;
f6) PORT=16437 STALE=3 ./rg.sh f6a $(PORT=16437 args l-f6) ;;
f7) PORT=16438 FAULT=4:BadRequest,9:ENOSPC ./rg.sh f7a $(PORT=16438 args l-f7) ;;
f8) PORT=16439 LINEFAULT=7:ValueError ./rg.sh f8a $(PORT=16439 args l-f8) ;;
f10) PORT=16435 ./rg.sh f10a $(PORT=16435 args l-xok) ;;
f11) PORT=16436 ./rg.sh f11a $(PORT=16436 args l-xbad) ;;
f9) PORT=16437 REPO=$S/repo2 ./rg.sh f9a $(PORT=16437 REPO=$S/repo2 args l-fl) --flow-decision $S/repo2/cpg/eval/decisions/rehearsal-flow-true.json --allow-after-adaptation ;;
esac
