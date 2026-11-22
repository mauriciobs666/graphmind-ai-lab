#!/usr/bin/env bash
# Add a committed ac5-gate-1 prereg record (in the COPY only), with values the copy itself observes.
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd $REPO/cpg/eval/tools
$PY - <<'P'
import json, gate as G
from pathlib import Path
repo = Path("../../..").resolve()
t = G.observe_tree(repo); o = G.engine_observed()
assert t["clean"], t["dirty"]
rec = {"schema": "ccg-prereg-v1", "codebase": {"name": "graphmind-ai-lab", "repo": "local", "tag": "rehearsal", "commit": t["head"]},
       "stratum": "ac5-gate-1", "kind": "gate", "packSetHash": t["packSetHash"], "coreCommit": t["head"], "coreTreeHash": t["coreTreeHash"],
       "algorithm": o["algorithm"], "defaultDepthBound": o["defaultDepthBound"], "firstContactDate": "2026-11-22", "recordedBy": "qa-engineer-rehearsal"}
with open(repo / "cpg/eval/preregistration.jsonl", "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
P
cd $REPO
git -c user.name=rehearsal -c user.email=rehearsal@invalid commit -q -m "rehearsal: ac5-gate-1 prereg record (copy only)" cpg/eval/preregistration.jsonl
git log --oneline -2
