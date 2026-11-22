#!/usr/bin/env bash
# state.sh <list-file>  -> marker state, answers lines, run.json present, torn-line check (counts and class names only; no ids)
. "$(dirname "$0")/env.sh"
K=$(HOME=$HOME_R $(dirname "$0")/key.sh "$1")
M=$CUST/gate-run-$K.marker; A=$CUST/run-$K.answers.jsonl; R=$CUST/run-$K.json
echo "key=${K:0:8} marker=$( [ -f $M ] && $PY -c "import json,sys;d=json.load(open('$M'));print(d['state'],'errorClass='+str(d.get('errorClass')),'resumes='+str(len(d.get('resumes',[]))),'stall='+str(d.get('stall')))" || echo none) answers=$( [ -f $A ] && wc -l < $A || echo 0) runjson=$( [ -f $R ] && echo yes || echo no) lastbyte=$( [ -f $A ] && tail -c1 $A | od -An -c | tr -d ' ' || echo -)"
