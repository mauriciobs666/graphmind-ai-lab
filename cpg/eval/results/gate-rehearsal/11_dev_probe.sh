#!/usr/bin/env bash
# 11_dev_probe.sh: the dev-gold probe (P6-1) as the PRODUCT of the F1 dev rehearsal (10_f1_run.sh, store $PORT still up): `gate.py dev-probe` scores F1's run.json by the gate's own
# path against the committed dev gold and writes $DEV_PROBE (record commit, harness sha map, gold hashes, snapshot id, hit ids, confirmed key-error ids).
# Every later `run` / `score` of a list that is not dev-only needs this exact file (`--dev-probe`; its sha256 is pinned in the marker). Exit 0 = every dev item of a gated stratum
# is a hit, a Q8 withheld shape (excused; the output prints the withheld counts per stratum and shape, counts only) or a stakeholder-confirmed key-error in
# cpg/eval/decisions/ccg-dev-key-errors-v1.json (read at the run's record commit), 1 = not: the gate would refuse to start.
. "$(dirname "$0")/env.sh"
L=$CUST/lists/dev-all.jsonl
KEY=$("$(dirname "$0")/key.sh" "$L")
( cd $REPO/cpg/eval/tools && HOME=$HOME_R $PY gate.py dev-probe --run $CUST/run-$KEY.json --out $DEV_PROBE --repo $REPO --system $SYSTEM --host 127.0.0.1 --port $PORT ) > $RH/log/dev-probe.out 2> $RH/log/dev-probe.err
echo "exit=$? $(cat $RH/log/dev-probe.out)" | tee $RH/log/dev-probe.rc
