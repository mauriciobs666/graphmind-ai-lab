#!/usr/bin/env python3
"""check_audit.py -- S11: header.sandboxAuditIds rule (bundle spec §2.2, egress §5.3): every id names a
``sandbox`` record with refused=false, exitCode=0 and egressMode equal to the bundle's.
  check_audit.py <bundle> <audit>     <audit> = the audit directory or its sandbox.jsonl; required, no default
                                      (a fixed default path checks the wrong file silently on another run dir)"""
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: check_audit.py <bundle> <audit dir | sandbox.jsonl>")
bundle = Path(sys.argv[1])
audit = Path(sys.argv[2])
audit_file = audit if audit.is_file() else audit / "sandbox.jsonl"
if not audit_file.is_file():
    sys.exit(f"check_audit: no such audit file: {audit_file}")
head = json.loads(bundle.read_bytes().split(b"\n")[0])
recs = {}
for line in audit_file.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    recs[r["auditId"]] = r
bad = []
for i in head["sandboxAuditIds"]:
    r = recs.get(i)
    if r is None or r.get("refused") or r.get("exitCode") != 0 or r.get("egressMode") != head["egressMode"]:
        bad.append((i, None if r is None else {k: r.get(k) for k in ("refused", "exitCode", "egressMode")}))
# the extract run's record ties to the reviewed bytes: header.tools.extractScriptHash == sandbox.script.sha256
ext = [recs[i] for i in head["sandboxAuditIds"] if i in recs and (recs[i].get("script") or {}).get("path") == "/scripts/extract.sc"]
if len(ext) != 1 or ext[0]["script"]["sha256"] != head["tools"]["extractScriptHash"]:
    bad.append(("extractScriptHash", [e.get("script") for e in ext]))
if bad:
    print("check_audit: FAIL", bad)
    sys.exit(1)
print(f"check_audit: {len(head['sandboxAuditIds'])} sandbox records ok (refused=false, exitCode=0, egressMode={head['egressMode']}); "
      "extractScriptHash == sandbox.script.sha256")
