#!/usr/bin/env python3
"""facts.py -- the harness's small JSON helper for run.sh (stdlib only; no network, no Docker).

    facts.py set  <facts.json> <key> <json>      facts[key] = <json>
    facts.py push <facts.json> <key> <json>      facts[key].append(<json>)
    facts.py snapshot                            the host's addresses and listening ports, as JSON, for the P-a probe:
                                                 {"v4": [...], "v6": [...], "tcp": [...], "udp": [...]}
                                                 v4/v6: every address of every host interface (`/usr/bin/ip -j -d address show`)
                                                 except 127.0.0.0/8 and ::1; tcp/udp: every listening port (`ss -H -ltn` / `-lun`)
"""
from __future__ import annotations

import ipaddress
import json
import subprocess
import sys
from pathlib import Path


def snapshot() -> dict:
    v4: list[str] = []
    v6: list[str] = []
    for ifc in json.loads(subprocess.run(["/usr/bin/ip", "-j", "-d", "address", "show"], capture_output=True, text=True,
                                         check=True, timeout=10).stdout):
        for a in ifc.get("addr_info", []):
            ip = ipaddress.ip_address(a["local"])
            if ip.is_loopback and (ip.version == 6 or ip in ipaddress.ip_network("127.0.0.0/8")):
                continue
            (v4 if ip.version == 4 else v6).append(str(ip))
    ports: dict[str, list[int]] = {}
    for proto, flag in (("tcp", "-ltnH"), ("udp", "-lunH")):
        out = subprocess.run(["ss", flag], capture_output=True, text=True, check=True, timeout=10).stdout
        found = set()
        for line in out.splitlines():
            cols = line.split()
            if len(cols) >= 4 and ":" in cols[3]:
                try:
                    found.add(int(cols[3].rsplit(":", 1)[1]))
                except ValueError:
                    pass
        ports[proto] = sorted(found)
    return {"v4": sorted(set(v4)), "v6": sorted(set(v6)), "tcp": ports["tcp"], "udp": ports["udp"]}


def main(argv: list[str]) -> int:
    if argv[:1] == ["snapshot"]:
        print(json.dumps(snapshot()))
        return 0
    if len(argv) == 4 and argv[0] in ("set", "push"):
        p = Path(argv[1])
        try:
            d = json.loads(p.read_text())
        except FileNotFoundError:
            d = {}
        except (OSError, ValueError) as exc:  # a facts file that does not parse is a failure, never a silent reset to {}
            print(f"facts.py: {p} is unreadable or not JSON: {exc}", file=sys.stderr)
            return 70
        val = json.loads(argv[3])
        if argv[0] == "set":
            d[argv[2]] = val
        else:
            d.setdefault(argv[2], []).append(val)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1) + "\n")
        tmp.replace(p)
        return 0
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
