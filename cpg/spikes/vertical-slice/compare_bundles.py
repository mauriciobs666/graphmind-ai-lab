#!/usr/bin/env python3
"""compare_bundles.py -- S11: SL-6's check.  Two bundles are equal iff they are byte-identical once the
two exempt fields (bundle spec §8.5: header.builtAt, header.sandboxAuditIds) are removed from line 1.
  compare_bundles.py a.bundle.jsonl b.bundle.jsonl      exit 0 identical, 1 different"""
import json
import sys


def stripped(path):
    data = open(path, "rb").read().split(b"\n")
    h = json.loads(data[0])
    for k in ("builtAt", "sandboxAuditIds"):
        h.pop(k)
    data[0] = json.dumps(h, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return data


a, b = stripped(sys.argv[1]), stripped(sys.argv[2])
if a == b:
    print(f"identical ({len(a) - 1} lines) apart from builtAt/sandboxAuditIds")
    sys.exit(0)
diff = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
print(f"DIFFERENT: lines {len(a) - 1} vs {len(b) - 1}; first differing line numbers {diff[:5]}")
sys.exit(1)
