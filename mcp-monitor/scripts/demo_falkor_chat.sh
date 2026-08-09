#!/usr/bin/env bash
set -euo pipefail

# demo_falkor_chat.sh — AC-3 manual runbook: a live, end-to-end demonstration
# that mcp-monitor detects an @mention posted into a real falkor-chat thread
# and launches its configured command.
#
# This is a RUNBOOK, not a pytest test (plan §8/§11): no falkor-chat server
# exists in the default/CI test environment, and AC-3 explicitly asks for a
# "demonstrated live, end to end" run against a real falkor-chat instance. QA
# (coordination doc unit 5) is expected to run this and record the result in
# the test report — it is not something this delivery executes on its own.
#
# What it does:
#   1. Checks falkor-chat's MCP endpoint is reachable (does NOT start
#      FalkorDB/falkor-chat for you — start those first, see below).
#   2. Writes a throwaway mcp-monitor config with one watch against
#      falkor-chat's read_messages tool, thread "demo-welcome" (the same
#      seeded demo thread kiro's falkor-chat-demo agent uses), pattern
#      '@mcp-monitor\b', repeat_trigger = false, launching a script that
#      appends to a marker file.
#   3. Starts mcp-monitor in the background against that config.
#   4. Posts a message into demo-welcome containing "@mcp-monitor" via
#      falkor-chat's own send_message MCP tool (using mcp-monitor's own
#      Python venv + the `mcp` SDK to speak MCP directly — no extra tooling).
#   5. Polls the marker file for up to ~30s for the triggered launch, prints
#      PASS/FAIL, and shows mcp-monitor's own log lines for the match.
#   6. Stops the background mcp-monitor process (falkor-chat is left running
#      — it was not started by this script).
#
# Prerequisites (start these yourself first, in another terminal or backgrounded):
#   ./falkor-chat/scripts/start_falkordb.sh -d
#   ./falkor-chat/scripts/start_server.sh          # REST + MCP on :8000 by default
#   (bootstrap_schema.sh + seed_demo.sh are both run by start_server.sh already)
#
# Usage:
#   mcp-monitor/scripts/demo_falkor_chat.sh
#
# Env overrides:
#   FALKORCHAT_MCP_URL   (default: http://localhost:8000/mcp)
#   DEMO_THREAD_ID        (default: demo-welcome — must match falkor-chat's seed)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MCP_URL="${FALKORCHAT_MCP_URL:-http://localhost:8000/mcp}"
THREAD_ID="${DEMO_THREAD_ID:-demo-welcome}"

if [ ! -x "$PY" ]; then
  echo "demo_falkor_chat.sh: no venv at $ROOT/.venv — run mcp-monitor/setup.sh first." >&2
  exit 1
fi

echo "── checking falkor-chat MCP endpoint at $MCP_URL ──────────"
if ! "$PY" - "$MCP_URL" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    # A GET against the MCP endpoint won't speak MCP, but a connection refused
    # vs. any HTTP response is enough to tell "server not running" from "up".
    urllib.request.urlopen(url, timeout=3)
except urllib.error.HTTPError:
    pass  # any HTTP response (even 4xx/5xx) means something is listening
except Exception as exc:
    print(f"unreachable: {exc}", file=sys.stderr)
    sys.exit(1)
PY
then
  cat >&2 <<EOF
demo_falkor_chat.sh: falkor-chat's MCP endpoint is not reachable at $MCP_URL.

Start it first:
  ./falkor-chat/scripts/start_falkordb.sh -d
  ./falkor-chat/scripts/start_server.sh
EOF
  exit 1
fi
echo "  reachable."

WORKDIR="$(mktemp -d)"
trap 'kill "${MONITOR_PID:-0}" 2>/dev/null || true; rm -rf "$WORKDIR"' EXIT

MARKER="$WORKDIR/marker.log"
: > "$MARKER"

LAUNCH_SCRIPT="$WORKDIR/on_trigger.py"
cat > "$LAUNCH_SCRIPT" <<PYEOF
import json, os, sys, time
payload = json.loads(sys.stdin.read())
with open(${MARKER@Q}, "a") as f:
    f.write(f"{time.time()!r} watch={payload['watch']} matched={payload['matched_text']!r}\n")
print(f"[demo] mcp-monitor triggered: {payload['matched_text']!r}", file=sys.stderr)
PYEOF

CONFIG="$WORKDIR/config.toml"
cat > "$CONFIG" <<TOMLEOF
[server.falkor-chat]
transport = "http"
url = "$MCP_URL"

[[watch]]
name = "falkor-chat-demo-mention"
server = "falkor-chat"
tool = "read_messages"
args = { re = "$THREAD_ID", since = 0, limit = 200 }
interval_seconds = 3
pattern = '@mcp-monitor\b'
repeat_trigger = false
command = ["$PY", "$LAUNCH_SCRIPT"]
TOMLEOF

LOG="$WORKDIR/mcp-monitor.log"
echo "── starting mcp-monitor (log: $LOG) ────────────────────"
"$ROOT/run.sh" --config "$CONFIG" --log-level INFO > "$LOG" 2>&1 &
MONITOR_PID=$!
sleep 1
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
  echo "demo_falkor_chat.sh: mcp-monitor exited immediately — see $LOG" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "  mcp-monitor running, pid=$MONITOR_PID"

STAMP="$(date +%s)"
BODY="@mcp-monitor please wake up (demo run $STAMP)"
echo "── posting message into thread '$THREAD_ID' ───────────────"
echo "  body: $BODY"
"$PY" - "$MCP_URL" "$THREAD_ID" "$BODY" <<'PY'
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

url, thread_id, body = sys.argv[1], sys.argv[2], sys.argv[3]


async def main() -> None:
    async with streamablehttp_client(url) as (read, write, *_rest):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "send_message", {"body": body, "re": thread_id}
            )
            if getattr(result, "isError", False):
                print(f"send_message reported an error: {result.content}", file=sys.stderr)
                raise SystemExit(1)


asyncio.run(main())
PY
echo "  posted."

echo "── waiting up to 30s for mcp-monitor to detect and trigger ─"
FOUND=0
for _ in $(seq 1 30); do
  if [ -s "$MARKER" ]; then
    FOUND=1
    break
  fi
  sleep 1
done

echo
if [ "$FOUND" -eq 1 ]; then
  echo "PASS — mcp-monitor detected the mention and launched the command:"
  cat "$MARKER"
  echo
  echo "mcp-monitor log excerpt:"
  grep -E "match found|command .* exited" "$LOG" || cat "$LOG"
  exit 0
else
  echo "FAIL — no trigger observed within 30s. mcp-monitor log:" >&2
  cat "$LOG" >&2
  exit 1
fi
