#!/usr/bin/env bash
# key.sh <list[,list]>  -> the list key (64 hex) the harness derives
. "$(dirname "$0")/env.sh"
cd $REPO/cpg/eval/tools && HOME=$HOME_R $PY /home/mauricio/prg/graphmind-ai-lab/cpg/eval/results/gate-rehearsal/key.py "$1"
