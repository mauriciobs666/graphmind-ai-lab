#!/usr/bin/env bash
# F1: the happy path on the whole dev list (flow flag false -> flow withheld). Records exit code, wall time, stdout, stderr.
. "$(dirname "$0")/env.sh"
L=$CUST/lists/dev-all.jsonl
t0=$(date +%s.%N)
run_gate --lists $L > $RH/log/f1.out 2> $RH/log/f1.err; rc=$?
t1=$(date +%s.%N)
echo "exit=$rc wall_s=$(echo "$t1 - $t0" | bc)" | tee $RH/log/f1.rc
