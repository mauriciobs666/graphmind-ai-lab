#!/usr/bin/env python3
"""coordgraph-hook-registered.py — is the coordination-graph post-edit hook registered?

Usage: coordgraph-hook-registered.py <repo-root> [<settings.json>...]

Aligned to `coordination-graph/coordgraph/hookreg.py` (the AUTHORITY for the user-facing sync-header claim)
and tested against the same golden table, `coordination-graph/tests/fixtures/hook-settings-cases.json`
(claude/scripts/coordgraph-hook-registered-selftest.sh). It is its own stdlib-only reading, never an import.

Settings files are given project first, local last (default: <root>/.claude/settings.json then
settings.local.json); missing or malformed ones are skipped, never fatal. For EACH of PostToolUse and
PostToolUseFailure independently, an entry counts when one of its handlers is `type: command` (a missing
`type` is command), has no `if` filter, and its command runs <root>/coordination-graph/hooks/post-tool-sync.sh:
as the executable (existing file with the exec bit) or as the argument of `bash`/`sh` (existing file), by
absolute path or $CLAUDE_PROJECT_DIR (a relative path is unproven). Echo/`bash -c`/env or timeout prefixes,
.bak suffixes, other checkouts and `..` detours do not count. The entry's matcher must cover Write, Edit and
Bash (empty/`*`; a name list split on `|` or `,`; otherwise a regex judged as a FULL match of the tool name,
an invalid one covers nothing). Tokens are unioned across entries and files. `disableAllHooks: true` (the
last file that sets the key wins) means not registered.

The kill switch (coordination-graph/.state/disabled) is NOT read here: it is an enablement state, not a
registration state; audit-team.sh check 15 reports it as a NOTE.

Prints one line per event: `<Event> ok` or `<Event> missing: <reason>` (plus `hooks disabled: ...`).
Exit 0 only when both events are ok and hooks are not disabled.
"""
import json
import os
import re
import shlex
import sys

SCRIPT = "coordination-graph/hooks/post-tool-sync.sh"
EVENTS = ("PostToolUse", "PostToolUseFailure")
NEED = ("Write", "Edit", "Bash")
_NAME_LIST = re.compile(r"[A-Za-z0-9_\-|, ]+")
_NAME_SEP = re.compile(r"[|,]")
_PROJECT_VAR = re.compile(r"\$\{CLAUDE_PROJECT_DIR\}|\$CLAUDE_PROJECT_DIR")


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.loads(fh.read())
    except (OSError, ValueError):
        return None


def is_script(root, command):
    try:
        tokens = shlex.split(_PROJECT_VAR.sub(root, command))
    except ValueError:
        return False
    shell = bool(tokens) and tokens[0] in ("bash", "sh")
    if shell:
        tokens = tokens[1:]
    # root is absolute, so a relative path (resolved against the session cwd, unproven) never matches
    if not tokens or os.path.normpath(tokens[0]) != os.path.normpath(os.path.join(root, SCRIPT)):
        return False
    return os.path.isfile(tokens[0]) if shell else (os.path.isfile(tokens[0]) and os.access(tokens[0], os.X_OK))


def tools_matched(matcher):
    if not isinstance(matcher, str):
        return set()
    if matcher in ("", "*"):
        return set(NEED)
    if _NAME_LIST.fullmatch(matcher):
        return {t.strip() for t in _NAME_SEP.split(matcher)} & set(NEED)
    try:
        pat = re.compile(matcher)
    except re.error:
        return set()
    return {t for t in NEED if pat.fullmatch(t)}


def main(argv):
    root = os.path.abspath(argv[0])
    files = argv[1:] or [os.path.join(root, ".claude", n) for n in ("settings.json", "settings.local.json")]
    seen = {e: {"cmd": False, "tokens": set()} for e in EVENTS}
    disabled = False
    for f in files:
        cfg = load(f)
        if not isinstance(cfg, dict):
            continue
        if isinstance(cfg.get("disableAllHooks"), bool):
            disabled = cfg["disableAllHooks"]
        hooks = cfg.get("hooks")
        if not isinstance(hooks, dict):
            continue
        for ev in EVENTS:
            entries = hooks.get(ev)
            for entry in entries if isinstance(entries, list) else []:
                inner = entry.get("hooks") if isinstance(entry, dict) else None
                if not isinstance(inner, list) or not any(
                        isinstance(h, dict) and h.get("type", "command") == "command" and "if" not in h
                        and is_script(root, str(h.get("command", ""))) for h in inner):
                    continue
                seen[ev]["cmd"] = True
                seen[ev]["tokens"] |= tools_matched(entry.get("matcher", ""))
    ok = True
    if disabled:
        print("hooks disabled: disableAllHooks is true (local file over project file)")
        ok = False
    for ev in EVENTS:
        s = seen[ev]
        if not s["cmd"]:
            print(f"{ev} missing: no valid hook entry runs {root}/{SCRIPT}")
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
    if len(sys.argv) < 2:
        print("usage: coordgraph-hook-registered.py <repo-root> [<settings.json>...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
