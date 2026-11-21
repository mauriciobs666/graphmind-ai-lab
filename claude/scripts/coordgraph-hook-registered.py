#!/usr/bin/env python3
"""coordgraph-hook-registered.py — is the coordination-graph post-edit hook registered?

Usage: coordgraph-hook-registered.py <command-needle> <settings.json>...

Reads the given Claude Code settings files (missing or malformed ones are skipped, never fatal) and, for
EACH of the events PostToolUse and PostToolUseFailure independently, looks for a hook entry whose
`command` contains <command-needle> and whose `matcher` covers Write, Edit and Bash (an exact-name list
split on `|`, or an empty / `*` matcher). Tokens are unioned across the entries of one event.

Prints one line per event: `<Event> ok` or `<Event> missing: <reason>`. Exit 0 only when both are ok.
Used by claude/scripts/audit-team.sh check 15; self-tested by coordgraph-hook-registered-selftest.sh.
"""
import json
import sys

EVENTS = ("PostToolUse", "PostToolUseFailure")
NEED = ("Write", "Edit", "Bash")


def main(argv):
    needle, files = argv[0], argv[1:]
    seen = {e: {"cmd": False, "tokens": set()} for e in EVENTS}
    for f in files:
        try:
            cfg = json.load(open(f))
        except Exception:
            continue
        hooks = cfg.get("hooks") if isinstance(cfg, dict) else None
        if not isinstance(hooks, dict):
            continue
        for ev in EVENTS:
            entries = hooks.get(ev)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                inner = entry.get("hooks")
                if not isinstance(inner, list):
                    continue
                if not any(isinstance(h, dict) and needle in str(h.get("command", "")) for h in inner):
                    continue
                seen[ev]["cmd"] = True
                m = entry.get("matcher", "")
                if not isinstance(m, str):
                    continue
                if m in ("", "*"):
                    seen[ev]["tokens"].update(NEED)
                else:
                    seen[ev]["tokens"].update(t.strip() for t in m.split("|"))
    ok = True
    for ev in EVENTS:
        s = seen[ev]
        if not s["cmd"]:
            print(f"{ev} missing: no hook entry runs {needle}")
            ok = False
            continue
        lack = [t for t in NEED if t not in s["tokens"]]
        if lack:
            print(f"{ev} missing: matcher does not cover {'/'.join(lack)}")
            ok = False
        else:
            print(f"{ev} ok")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: coordgraph-hook-registered.py <command-needle> <settings.json>...", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
