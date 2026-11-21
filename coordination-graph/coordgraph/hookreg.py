"""Is the post-edit hook registered? The sync header's `Hook:` state (README "Reading the sync header").

`registered` means the project's `.claude/settings.json` / `settings.local.json` register
`coordination-graph/hooks/post-tool-sync.sh` for BOTH `PostToolUse` and `PostToolUseFailure` with a matcher
covering Write, Edit and Bash; `disabled` (kill switch) wins over everything; anything unreadable, garbled or
unproven is `not registered`: the header never over-claims. This is a static reading of the files: it does not
know whether Claude Code actually fires the hook in a given context (docs/hook-settings.md section 6).

Matchers follow Claude Code: empty or `*` matches every tool, a name list of letters/digits/`_` split on `|` is
exact, anything else is a regular expression, judged conservatively as a full match of the tool name
(`Wri.|Edi.|Bas.` is not enough; an invalid one covers nothing). The handler
must be `type: command` with no `if` filter, and its command must name this script as its executable (an
existing file with the executable bit) or as the argument of `bash`/`sh` (an existing file), by an absolute
path or `$CLAUDE_PROJECT_DIR` (a relative path resolves against the session cwd, so it is unproven), so `.bak`
suffixes, `echo <path>`, `bash -c`, env/`timeout` prefixes, another checkout and `..` detours do not count.
`disableAllHooks: true` (local file over project file) means `not registered`. Not readable from the project
and therefore not judged: user-scope settings (`~/.claude/settings.json`) and managed policy
(`disableAllHooks`, `allowManagedHooksOnly`). Deliberately its own reading, not an import of `claude/scripts/coordgraph-hook-registered.py`.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

from .config import Config

SCRIPT = "coordination-graph/hooks/post-tool-sync.sh"
EVENTS = ("PostToolUse", "PostToolUseFailure")
TOOLS = ("Write", "Edit", "Bash")
SETTINGS = ("settings.json", "settings.local.json")
_NAME_LIST = re.compile(r"[A-Za-z0-9_\-|, ]+")
_NAME_SEP = re.compile(r"[|,]")
_PROJECT_VAR = re.compile(r"\$\{CLAUDE_PROJECT_DIR\}|\$CLAUDE_PROJECT_DIR")


def hook_status(cfg: Config, root: Path) -> str:
    """`disabled` | `registered` | `not registered`."""
    if cfg.hook_disabled or (cfg.state_dir / "disabled").exists():
        return "disabled"
    root = Path(os.path.abspath(root))
    if _all_hooks_disabled(root):
        return "not registered"
    covered = {ev: set() for ev in EVENTS}
    for name in SETTINGS:
        for ev, tools in _covered_by(root, root / ".claude" / name).items():
            covered[ev] |= tools
    return "registered" if all(set(TOOLS) <= covered[ev] for ev in EVENTS) else "not registered"


def _all_hooks_disabled(root: Path) -> bool:
    """`disableAllHooks` as Claude Code resolves it: the local file over the project file (user and managed
    scopes are not readable from here)."""
    value = False
    for name in SETTINGS:  # project first, local last: the later file wins when it sets the key
        try:
            cfg = json.loads((root / ".claude" / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("disableAllHooks"), bool):
            value = cfg["disableAllHooks"]
    return value


def _covered_by(root: Path, path: Path) -> dict[str, set[str]]:
    out = {ev: set() for ev in EVENTS}
    try:
        hooks = json.loads(path.read_text(encoding="utf-8")).get("hooks")
    except (OSError, ValueError, AttributeError):
        return out
    if not isinstance(hooks, dict):
        return out
    for ev in EVENTS:
        entries = hooks.get(ev)
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict) or not _runs_script(root, entry.get("hooks")):
                continue
            out[ev] |= _tools_matched(entry.get("matcher", ""))
    return out


def _runs_script(root: Path, inner) -> bool:
    return isinstance(inner, list) and any(
        isinstance(h, dict) and h.get("type", "command") == "command" and "if" not in h
        and _is_script(root, str(h.get("command", ""))) for h in inner)


def _is_script(root: Path, command: str) -> bool:
    expanded = _PROJECT_VAR.sub(str(root), command)
    try:
        tokens = shlex.split(expanded)
    except ValueError:
        return False
    shell = bool(tokens) and tokens[0] in ("bash", "sh")
    if shell:
        tokens = tokens[1:]
    # `root` is absolute, so a relative path (resolved against the session cwd, unproven) never equals it
    if not tokens or os.path.normpath(tokens[0]) != os.path.normpath(root / SCRIPT):
        return False
    return os.path.isfile(tokens[0]) if shell else (os.path.isfile(tokens[0]) and os.access(tokens[0], os.X_OK))


def _tools_matched(matcher) -> set[str]:
    if not isinstance(matcher, str):
        return set()
    if matcher in ("", "*"):
        return set(TOOLS)
    if _NAME_LIST.fullmatch(matcher):
        return {t.strip() for t in _NAME_SEP.split(matcher)} & set(TOOLS)
    try:
        pattern = re.compile(matcher)
    except re.error:
        return set()
    return {t for t in TOOLS if pattern.fullmatch(t)}
