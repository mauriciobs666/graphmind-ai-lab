#!/usr/bin/env bash
# Item 1 setup: real ACL users on the throwaway (private CCG_CREDENTIALS_DIR), dev list + probe map under the scratch custodian.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p $CUST/lists $CCG_CREDENTIALS_DIR; chmod 700 $CUST $CUST/lists $CCG_CREDENTIALS_DIR
cd $REPO/cpg/eval/tools
HOME=$HOME_R $PY - <<'P'
import os, json, sys
from pathlib import Path
sys.path.insert(0, "..")
import gate as G
from ccg.store import acl, loader as L
pw = None   # nopass rehearsal store (see 00_store.sh)
admin = L.connect("127.0.0.1", int(os.environ["PORT"]), username=None, password=pw)
users = acl.create_users(admin, os.environ["SYSTEM"])
print("acl users created:", len(users), sorted(u.rsplit("_", 1)[0] for u in users) if False else "")
cust = Path(os.environ["CUST"]); dev = Path(os.environ["REAL"]) / "cpg/eval/dev"
allrecs = []
for k in ("http", "mcp", "scripts", "wiring", "flow"):
    allrecs += [json.loads(l) for l in (dev / f"ac5-{k}.jsonl").read_text().splitlines() if l.strip()]
(cust / "lists" / "dev-all.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in allrecs))
probes = {r["anchorId"]: G.unit_probe(r, allrecs) for r in allrecs if r["anchor"].get("type") == "unit"}
(cust / "s47-probe-map.json").write_text(json.dumps(probes))
print("records", len(allrecs), "unit probes", len(probes))
P
ls -l $CCG_CREDENTIALS_DIR | awk '{print $1,$9}'
