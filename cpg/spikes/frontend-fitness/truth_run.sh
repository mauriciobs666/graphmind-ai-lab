#!/usr/bin/env bash
# truth_run.sh -- S25 reference-truth run for ONE language, in the `ccg-toolchain-truth` image, with
# network, OUTSIDE the sandbox and never in a fetch-off Joern workspace (ML §9, §11 N1).
#
#   truth_run.sh <lang> [--image <ref>] [--mem <size>] [--env K=V]...
#
# Reads  cpg/.ccg-work/corpora/<lang>/src            (read-only mount; the pinned, clean tree)
# Works in cpg/.ccg-work/frontend-fitness/truth-work/<lang>/   (a writable COPY + dependency caches: this
#         directory is the only place a venv / node_modules / ~/.m2 / ~/.nuget / GOMODCACHE ever exists)
# Writes cpg/.ccg-work/frontend-fitness/truth-out/<lang>/{index.scip,index.json,callgraph.txt,run.log,run.meta}
# Prints a one-line JSON with exit status and wall seconds on stdout. Never touches any other container.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPG="$(cd "$HERE/../.." && pwd)"
LANG_="${1:?usage: truth_run.sh <lang> [--image ref] [--mem size]}"; shift
IMAGE="" MEM=3g ENVS=()
while [ $# -gt 0 ]; do case "$1" in --image) IMAGE="$2"; shift 2;; --mem) MEM="$2"; shift 2;; --env) ENVS+=(-e "$2"); shift 2;; *) echo "bad arg $1" >&2; exit 64;; esac; done
[ -n "$IMAGE" ] || IMAGE="$("$CPG/infra/toolchain/build.sh" --truth --print-tag)"
SRC="$CPG/.ccg-work/corpora/$LANG_/src"
W="$CPG/.ccg-work/frontend-fitness/truth-work/$LANG_"
O="$CPG/.ccg-work/frontend-fitness/truth-out/$LANG_"
[ -d "$SRC" ] || { echo "no corpus at $SRC (run_fitness.py fetch $LANG_)" >&2; exit 66; }
mkdir -p "$W" "$O"
NAME="ccg-s25-truth-$LANG_-$$"
START=$(date +%s)

# the in-container script (heredoc below) is run by `bash -l`; /corpus is read-only, /work writable, /out outputs
case "$LANG_" in
python) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  uv venv --seed --python python3 /work/venv >/dev/null
  uv pip install --python /work/venv/bin/python -e ".[dev]" >/work/pip.log 2>&1
  export VIRTUAL_ENV=/work/venv PATH=/work/venv/bin:$PATH
  scip-python index --cwd /work/src --project-name falkorchat --project-version 1 --output /out/index.scip
  ' ;;
javascript) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  npm ci --ignore-scripts --no-audit --no-fund >/work/npm.log 2>&1
  # tests/ (Playwright e2e) is in no tsconfig of the corpus: give the truth tool a third project over it, in the
  # WORK COPY only (the corpus tree and the labeller never see this file)
  printf "%s" "{\"extends\":\"./tsconfig.app.json\",\"compilerOptions\":{\"types\":[\"node\"],\"noUnusedLocals\":false,\"noUnusedParameters\":false},\"include\":[\"tests\"]}" >tsconfig.fitness-tests.json
  scip-typescript index --cwd /work/src --no-progress-bar --output /out/index.scip tsconfig.app.json tsconfig.node.json tsconfig.fitness-tests.json
  ' ;;
go) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  export GOFLAGS=-mod=mod GOCACHE=/work/gocache GOPATH=/work/gopath GOTOOLCHAIN=local
  go mod download >/work/go-mod.log 2>&1
  callgraph -algo=vta -test -format="{{.Caller}} -> {{.Callee}} @ {{.Filename}}:{{.Line}}:{{.Column}}" ./... >/out/callgraph.txt
  go list -m >/out/module.txt
  ' ;;
java) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  bash /corpus-truth/java.sh
  ' ;;
kotlin) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  bash /corpus-truth/kotlin.sh
  ' ;;
csharp) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  export DOTNET_CLI_HOME=/work/dotnet NUGET_PACKAGES=/work/nuget DOTNET_NOLOGO=1
  slns="$(ls *.sln *.slnx 2>/dev/null | tr "\n" " ")"
  scip-dotnet index $slns --working-directory /work/src --output /out/index.scip
  ' ;;
cpp) BODY='
  cp -a /corpus/. /work/src/ && cd /work/src
  bash /corpus-truth/cpp.sh
  ' ;;
*) echo "unknown lang" >&2; exit 64;; esac

rm -rf "$W"/* "$O"/* 2>/dev/null || true
mkdir -p "$W" "$O"
set +e
docker run --rm --name "$NAME" --memory "$MEM" --cpus 2 --hostname truth \
  -v "$SRC":/corpus:ro -v "$HERE/truth-scripts":/corpus-truth:ro -v "$W":/work -v "$O":/out \
  -e HOME=/work/home -e TMPDIR=/work/tmp "${ENVS[@]}" -w /work "$IMAGE" \
  bash -lc "mkdir -p /work/home /work/tmp /work/src; set -e; $BODY
    if [ -f /out/index.scip ]; then scip print --json /out/index.scip >/out/index.json; fi" >"$O/run.log" 2>&1
RC=$?
set -e
END=$(date +%s)
printf '{"lang":"%s","image":"%s","exit":%d,"wallSeconds":%d,"startedEpoch":%d}\n' "$LANG_" "$IMAGE" "$RC" "$((END-START))" "$START" | tee "$O/run.meta"
exit 0
