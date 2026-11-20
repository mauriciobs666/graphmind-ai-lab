#!/usr/bin/env bash
# TEMPORARY S0 probe (coordination-graph plan, S0). Remove after the run; result goes to
# skills/agent-standards/claude-code.md (Hooks). Runbook: claude/cobb/scripts/hook-probe.md.
# Registered on PostToolUse and PostToolUseFailure (matcher Write|Edit|Bash). Appends one JSON
# line per firing to $COORDGRAPH_PROBE_LOG (default /tmp/coordgraph-hook-probe.log).
# Observe-only: exit 0 always, except the explicit exit-2 leg (see below). Never blocks work.
# Exit-2 leg: when the tool input contains the literal marker PROBE_EXIT2 the script writes a
# sentence to stderr and exits 2, so the runbook can check whether the agent sees that stderr.
set -u
LOG="${COORDGRAPH_PROBE_LOG:-/tmp/coordgraph-hook-probe.log}"
PAYLOAD="$(cat)"
export PAYLOAD LOG
python3 -I - <<'PY'
import json, os, time
raw = os.environ.get("PAYLOAD", "")
try:
    p = json.loads(raw)
except Exception:
    p = {"_unparsed": raw[:200]}
ti = p.get("tool_input") or {}
rec = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "event": p.get("hook_event_name"),
    "tool": p.get("tool_name"),
    "agent_id": p.get("agent_id"),
    "agent_type": p.get("agent_type"),
    "session_id": p.get("session_id"),
    "permission_mode": p.get("permission_mode"),
    "cwd": p.get("cwd"),
    "env_CLAUDE_PROJECT_DIR": os.environ.get("CLAUDE_PROJECT_DIR"),
    "target": ti.get("file_path") or (ti.get("command") or "")[:160],
    "payload_keys": sorted(p.keys()),
}
for k in ("tool_response", "error", "is_interrupt"):
    if k in p:
        v = p[k]
        rec[k] = (json.dumps(v)[:200] if not isinstance(v, str) else v[:200])
with open(os.environ["LOG"], "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
case "$PAYLOAD" in
  *PROBE_EXIT2*)
    echo "HOOK-PROBE-EXIT2-SENTINEL: if you can read this sentence, hook stderr reached the agent." >&2
    exit 2 ;;
esac
exit 0
