"""Is the post-edit hook registered? The sync header's `Hook:` state (README "Reading the sync header").

`registered` means the project's `.claude/settings.json` / `settings.local.json` register
`coordination-graph/hooks/post-tool-sync.sh` for BOTH `PostToolUse` and `PostToolUseFailure` with a matcher
covering Write, Edit and Bash; `disabled` (kill switch) wins over everything; anything unreadable, garbled or
unproven is `not registered`: the header never over-claims. This is a static reading of the files: it does not
know whether Claude Code actually fires the hook in a given context (docs/hook-settings.md section 6).

Matchers follow Claude Code: empty or `*` matches every tool, a name list of letters/digits/`_` split on `|` is
exact, anything else is a regular expression, judged conservatively as a full match of the tool name
(`Wri.|Edi.|Bas.` is not enough; an invalid one covers nothing). The command
must name this script as its executable, or as the argument of `bash`/`sh`, resolved against the repo root
with `$CLAUDE_PROJECT_DIR` expanded, so `.bak` suffixes, `echo <path>`, another checkout and `..` detours do
not count. Deliberately its own reading, not an import of `claude/scripts/coordgraph-hook-registered.py`.
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
_NAME_LIST = re.compile(r"[A-Za-z0-9_|]+")
_PROJECT_VAR = re.compile(r"\$\{CLAUDE_PROJECT_DIR\}|\$CLAUDE_PROJECT_DIR")


def hook_status(cfg: Config, root: Path) -> str:
    """`disabled` | `registered` | `not registered`."""
    if cfg.hook_disabled or (cfg.state_dir / "disabled").exists():
        return "disabled"
    root = Path(root)
    covered = {ev: set() for ev in EVENTS}
    for name in SETTINGS:
        for ev, tools in _covered_by(root, root / ".claude" / name).items():
            covered[ev] |= tools
    return "registered" if all(set(TOOLS) <= covered[ev] for ev in EVENTS) else "not registered"


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
        isinstance(h, dict) and _is_script(root, str(h.get("command", ""))) for h in inner)


def _is_script(root: Path, command: str) -> bool:
    expanded = _PROJECT_VAR.sub(str(root), command)
    try:
        tokens = shlex.split(expanded)
    except ValueError:
        return False
    if tokens and tokens[0] in ("bash", "sh"):
        tokens = tokens[1:]
    return bool(tokens) and os.path.normpath(root / tokens[0]) == os.path.normpath(root / SCRIPT)


def _tools_matched(matcher) -> set[str]:
    if not isinstance(matcher, str):
        return set()
    if matcher in ("", "*"):
        return set(TOOLS)
    if _NAME_LIST.fullmatch(matcher):
        return set(matcher.split("|")) & set(TOOLS)
    try:
        pattern = re.compile(matcher)
    except re.error:
        return set()
    return {t for t in TOOLS if pattern.fullmatch(t)}
