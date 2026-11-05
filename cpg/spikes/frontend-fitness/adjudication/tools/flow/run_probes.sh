#!/usr/bin/env bash
# Run one probe script on a COPY of a language's S27 off CPG, through the sandbox runner (offline), into $FLOW_WORK.
#   cd cpg && FLOW_WORK=/tmp/ccg-flow-adj spikes/frontend-fitness/adjudication/tools/flow/run_probes.sh <lang> rdrelax|rdfix|rdwide
# rdrelax -> $FLOW_WORK/run/<lang>/ (records d = shipped walk, d2 = R1); rdfix -> $FLOW_WORK/runfix/<lang>/ (d4 = R2).
# rdwide (every reaching-def source, upper bound) ran out of Java heap on python, go, csharp and cpp.
# cdgprobe.sc / idprobe.sc take --param targets=...; run them the same way with their own params (see the files' headers).
set -euo pipefail
lang=$1; probe=$2
CPG=$(cd "$(dirname "$0")/../../../../.." && pwd)                       # cpg/
WORK=${FLOW_WORK:-/tmp/ccg-flow-adj}
J=$CPG/.ccg-work/frontend-fitness/joern/$lang
case $probe in rdfix) out=$WORK/runfix/$lang ;; *) out=$WORK/run/$lang ;; esac
mkdir -p "$out"
cpg=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['runs']['off.parse']['cpgFile'])" "$J/s27-run.json")
n=$(basename "$cpg"); [ -f "$out/$n" ] || cp "$J/off/$n" "$out/"
"$CPG/sandbox/offline-run.sh" --backend container --mode offline --frontend joern \
  --script-file "$(dirname "$0")/$probe.sc" --source "$J/stage" --output "$out" -- \
  --script "/scripts/$probe.sc" --param "cpgFile=/out/$n" --param "out=/out/@AUDIT_ID@.facts.jsonl" --param "lang=$lang"
