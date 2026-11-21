#!/usr/bin/env bash
# U12 memory/latency proof: throwaway container u12-hybrid on 127.0.0.1:16612 only.
# Never touches falkordb-dev. QUERY_MEM_CAPACITY is set on the throwaway only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; SERVER="$HERE/../../../../server"
docker rm -f u12-hybrid >/dev/null 2>&1 || true
docker run -d --name u12-hybrid -p 127.0.0.1:16612:6379 falkordb/falkordb:v4.18.11 >/dev/null
for _ in $(seq 60); do redis-cli -p 16612 PING 2>/dev/null | grep -q PONG && break; sleep 0.5; done
echo "container: $(docker inspect -f '{{.Name}} {{.Config.Image}}' u12-hybrid)"
"$SERVER/.venv/bin/python" "$HERE/live_proof.py" 16612
docker rm -f u12-hybrid >/dev/null
echo "container removed"
