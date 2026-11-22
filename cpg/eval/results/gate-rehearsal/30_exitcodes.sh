#!/usr/bin/env bash
# Item 7: cheap exit-code probes against the spent F1 list (store 16432) and a record without `kind` (repo3). Each: real process, exit code, stdout/stderr bytes, leak scan.
. "$(dirname "$0")/env.sh"
D=$(dirname "$(readlink -f "$0")")
probe() { # tag, repo, extra args...
  tag=$1; repo=$2; shift 2
  ( cd $repo/cpg/eval/tools && REPO=$repo HOME=$HOME_R $PY $D/gr.py run --custodian $CUST --repo $repo --system $SYSTEM --host 127.0.0.1 --port $PORT "$@" ) > $RH/log/$tag.out 2> $RH/log/$tag.err
  echo "$tag exit=$? stdout=$(wc -c < $RH/log/$tag.out)B stderr=$(wc -c < $RH/log/$tag.err)B :: $(head -c 160 $RH/log/$tag.err | tr '\n' ' ')"
}
L=$CUST/lists/dev-all.jsonl
probe x-rerun-spent      $REPO  --gate-id $GATE_ID --lists $L                         # expect 2: already run
probe x-resume-asked     $REPO  --gate-id $GATE_ID --lists $L --resume                # expect 2: resume of an asked run
probe x-missing-list     $REPO  --gate-id $GATE_ID --lists $CUST/lists/nope.jsonl      # expect 4: bad input
probe x-outside-list     $REPO  --gate-id $GATE_ID --lists $REAL/cpg/eval/dev/ac5-http.jsonl   # expect 2: list outside the custodian
probe x-no-kind-record   $S/repo3 --gate-id ac5-gate-nokind --lists $L                # expect 5: P3-2 precheck refusal, nothing spent
probe x-no-record        $REPO  --gate-id ac5-gate-none --lists $L                    # expect 2: no record
probe x-resume-nomarker  $REPO  --gate-id $GATE_ID --lists $CUST/lists/l-f7.jsonl --resume   # expect 2: nothing to resume
ls $CUST | grep -c 'nokind' || true
