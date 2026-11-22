"""Scenario lists under the scratch custodian: dev-all minus one record each (a different list key per scenario; dev records only)."""
import json, os
from pathlib import Path
cust = Path(os.environ["CUST"]); src = [json.loads(l) for l in (cust / "lists" / "dev-all.jsonl").read_text().splitlines()]
non_flow = [i for i, r in enumerate(src) if r["derivedStratum"] != "flow"]
for n, name in enumerate(["f2", "f3", "f4", "f5", "f6", "f7", "fl", "f8"], start=1):
    keep = [r for i, r in enumerate(src) if i != non_flow[n * 3]]
    (cust / "lists" / f"l-{name}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep))
    print(name, len(keep))
