#!/usr/bin/env python3
"""The campaign's one guard: a `PreToolUse` hook, the only bound on a tool call
(`docs/plans/typed-decision-models6.md` §3.3 over plan4 §3.4).

Installed as `exchange/.claude/guard_hook.py` and wired once, in `exchange/.claude/settings.json`
(matcher `*`, `"onFailure": "block"`); no agent definition carries `hooks`. Settings hooks fire in
the driver and in both subagents, keyed on the input's `agent_type`. Stdlib only, and it imports
nothing from `decisionmodels`: the audit's bounds code is separate, so one defect cannot hide in
both (plan6 §3.4).

**Decisions** are printed as `{"hookSpecificOutput": {"hookEventName": "PreToolUse",
"permissionDecision": "allow"|"deny", "permissionDecisionReason": ...}}` and exit 0:
- `dm-driver` may call only `Agent`: `subagent_type` `dm-generator` or `dm-labeler` (the one that
  matches the `role` in the header of the single request in `exchange/req/`), keys limited to
  `subagent_type`, `prompt` and `description` (so no `model`, `effort`, `isolation`,
  `run_in_background`), and a `prompt` byte-equal to that role's installed spawn template with the
  request's absolute path;
- `dm-generator`/`dm-labeler` may call `Read` strictly inside `exchange/req/`, `Write` strictly
  inside `exchange/ans/`, and `SubagentHandback` (whether a hook ever sees it is for U5.7's smoke
  to record); any other tool is denied;
- **paths are canonical**: `file_path` must be a non-empty string; a relative one is joined to the
  input's `cwd`; the result goes through `os.path.realpath` (symlinks, `..`) and must lie strictly
  inside `realpath(exchange/req)` or `realpath(exchange/ans)`: the folder itself and a sibling
  with the same prefix (`ans-x/`) are outside.

**Fail closed.** A missing or unknown `agent_type`, an event other than `PreToolUse`, a missing
`tool_name` or `tool_input`, unparseable input, or any exception exits 1 with no decision on
stdout and the reason on stderr; the wiring's `onFailure: "block"` then blocks the call. Every
decision, whatever it is, is appended to `store/hook-log.jsonl` first (a failed append is such an
exception), with `agent_type`, the tool-use id and the decision, never a path or a prompt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DRIVER = "dm-driver"
SUBAGENTS = ("dm-generator", "dm-labeler")
AGENT_FOR_ROLE = {"generate": "dm-generator", "label": "dm-labeler"}
TEMPLATE_FILES = {"dm-generator": "spawn-generator.txt", "dm-labeler": "spawn-labeler.txt"}
AGENT_KEYS = frozenset({"subagent_type", "prompt", "description"})
HANDBACK = "SubagentHandback"


class Failure(Exception):
    """A guard failure: exit 1 with no decision, so `onFailure: "block"` blocks the call."""


def exchange_dir() -> Path:
    """`<campaign>/exchange`, from where this file is installed: `exchange/.claude/`."""
    return Path(os.path.realpath(__file__)).parent.parent


def _inside(path: Any, cwd: Any, folder: Path) -> bool:
    """Whether `path` (string, relative to `cwd`) resolves strictly inside `folder`."""
    if not isinstance(path, str) or not path:
        return False
    try:
        if not os.path.isabs(path):
            if not isinstance(cwd, str) or not os.path.isabs(cwd):
                return False
            path = os.path.join(cwd, path)
        resolved = os.path.realpath(path)
        bound = os.path.realpath(folder)
    except ValueError:             # an embedded NUL
        return False
    return resolved.startswith(bound + os.sep)


def _pending_request(exchange: Path) -> Path | None:
    """The one request in `exchange/req/` (the driver writes one at a time), else `None`."""
    found = sorted((exchange / "req").glob("*.jsonl"))
    return found[0] if len(found) == 1 else None


def _agent_decision(tool_input: dict[str, Any], exchange: Path) -> tuple[str, str]:
    extra = sorted(set(tool_input) - AGENT_KEYS)
    if extra:
        return "deny", f"Agent input has keys outside the allowed set: {extra}"
    subagent, prompt, description = (tool_input.get(k) for k in
                                     ("subagent_type", "prompt", "description"))
    if subagent not in SUBAGENTS:
        return "deny", "Agent subagent_type is not dm-generator or dm-labeler"
    if not isinstance(prompt, str):
        return "deny", "Agent prompt is not a string"
    if description is not None and not isinstance(description, str):
        return "deny", "Agent description is not a string"
    request = _pending_request(exchange)
    if request is None:
        return "deny", "exchange/req/ does not hold exactly one request"
    with open(request, encoding="utf-8") as handle:
        role = json.loads(handle.readline()).get("role")
    if AGENT_FOR_ROLE.get(role) != subagent:
        return "deny", "Agent subagent_type is not the pending request's role"
    template = (exchange / ".claude" / TEMPLATE_FILES[subagent]).read_text("utf-8")
    expected = template.replace("{request_path}", str(request))
    if prompt != expected:
        return "deny", "Agent prompt is not the pending spawn prompt"
    return "allow", "the pending spawn"


def decide(data: Any, exchange: Path) -> tuple[str, str]:
    """`(decision, reason)` for one hook input; `Failure` where the input cannot be judged."""
    if not isinstance(data, dict):
        raise Failure("the hook input is not a JSON object")
    if data.get("hook_event_name") != "PreToolUse":
        raise Failure("the hook event is not PreToolUse")
    agent = data.get("agent_type")
    if agent != DRIVER and agent not in SUBAGENTS:
        raise Failure(f"unknown or missing agent_type: {agent!r}")
    tool, tool_input = data.get("tool_name"), data.get("tool_input")
    if not isinstance(tool, str) or not isinstance(tool_input, dict):
        raise Failure("tool_name or tool_input is missing")
    if agent == DRIVER:
        if tool != "Agent":
            return "deny", f"dm-driver may call only Agent, not {tool}"
        return _agent_decision(tool_input, exchange)
    if tool == HANDBACK:
        return "allow", "the subagent's hand-back"
    folder = {"Read": "req", "Write": "ans"}.get(tool)
    if folder is None:
        return "deny", f"{agent} may call only Read and Write, not {tool}"
    if _inside(tool_input.get("file_path"), data.get("cwd"), exchange / folder):
        return "allow", f"{tool} inside exchange/{folder}/"
    return "deny", f"{tool} outside exchange/{folder}/"


def _log(data: dict[str, Any], decision: str, exchange: Path) -> None:
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "event": data["hook_event_name"], "agent_type": data["agent_type"],
              "tool_name": data["tool_name"], "tool_use_id": data.get("tool_use_id"),
              "decision": decision}
    with open(exchange.parent / "store" / "hook-log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> int:
    try:
        data = json.loads(sys.stdin.read())
        exchange = exchange_dir()
        decision, reason = decide(data, exchange)
        _log(data, decision, exchange)
    except Exception as exc:  # fail closed, whatever it is
        print(f"guard failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": decision,
        "permissionDecisionReason": reason}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
