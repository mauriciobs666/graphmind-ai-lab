#!/usr/bin/env bash
set -euo pipefail

# handle_trigger.sh — minimal example [[watch]] launch command, wired up by
# config.example.toml's `fake-server-demo` watch.
#
# It exists to make FR-5 (a launched command receives the full match payload
# as JSON on stdin, plus MCP_MONITOR_WATCH_NAME/_SERVER_NAME/_TOOL_NAME as
# env-var convenience) concretely observable to someone reading the example,
# by reading that payload and printing it. This is a documentation example,
# not an automated test's assertion target, so stderr output is enough — no
# marker file (compare scripts/demo_falkor_chat.sh's inline on_trigger.py,
# which does write one, because that script's automated PASS/FAIL check needs
# something to poll for).
#
# A real watch's command would do something more useful with the payload
# (wake a headless agent, page someone, open a ticket, ...) — see README.md
# for the full stdin JSON schema.

payload="$(cat)"

echo "[handle_trigger.sh] watch=${MCP_MONITOR_WATCH_NAME:-?} server=${MCP_MONITOR_SERVER_NAME:-?} tool=${MCP_MONITOR_TOOL_NAME:-?}" >&2
echo "[handle_trigger.sh] payload: $payload" >&2
