#!/usr/bin/env bash
# Throwaway container u12-f1 on 127.0.0.1:16613 only; removed afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker rm -f u12-f1 >/dev/null 2>&1 || true
docker run -d --name u12-f1 -p 127.0.0.1:16613:6379 falkordb/falkordb:v4.18.11 >/dev/null
for _ in $(seq 60); do redis-cli -p 16613 PING 2>/dev/null | grep -q PONG && break; sleep 0.5; done
"$HERE/../../../../server/.venv/bin/python" "$HERE/f1_barrier_probe.py" 16613
docker rm -f u12-f1 >/dev/null; echo "container removed"
