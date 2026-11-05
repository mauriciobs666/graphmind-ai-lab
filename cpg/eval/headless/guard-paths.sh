#!/bin/sh
# guard-paths.sh — PreToolUse path guard for the headless ground-truth writers (S52, P0-43).
# docs/plans/connected-code-graph-p0-2.md §3 S52, §6.3.
#
# Contract (Claude Code hook): the hook JSON arrives on stdin (tool_name, tool_input, cwd).
#   exit 0                  allow (no output)
#   exit 2 + stderr reason  deny; the reason goes back to the model
# Fail-closed: unreadable or malformed input, a missing path on a path-bearing tool, or any
# internal error denies. Only exit 0 allows.
#
# Rules (root = the session's cwd, symlinks resolved):
#   1. Every path a file tool names (file_path, notebook_path, path), made absolute against the
#      root and fully resolved (".." and symlinks), must be the root or inside it.
#   2. A write (Write, Edit, NotebookEdit) must resolve strictly inside <root>/out/.
#   3. A Glob pattern or Grep glob that is absolute, starts with "~", or has a ".." component
#      is denied: it would walk out of the root without a path argument.
#   4. A tool this guard does not know and that carries no path field is not its concern (the
#      permission layer owns tool availability); one that carries a path field is checked.
# Requires /usr/bin/python3 (the launcher's PATH is /usr/bin:/bin).
PY=/usr/bin/python3
[ -x "$PY" ] || PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || { echo "guard-paths: python3 not found; denying" >&2; exit 2; }
exec "$PY" -I -c '
import json, os, sys

def deny(msg):
    sys.stderr.write("guard-paths: " + msg + "\n")
    sys.exit(2)

try:
    data = json.loads(sys.stdin.read())
    if not isinstance(data, dict):
        raise ValueError("hook input is not an object")
    tool = data.get("tool_name")
    inp = data.get("tool_input")
    if not isinstance(tool, str) or not isinstance(inp, dict):
        raise ValueError("tool_name/tool_input missing")
    cwd = data.get("cwd")
    if cwd is None:
        cwd = os.getcwd()
    if not isinstance(cwd, str) or not cwd or "\0" in cwd:
        raise ValueError("bad cwd")
    root = os.path.realpath(cwd)
except Exception as exc:
    deny("unreadable hook input (%s)" % exc)

WRITE = {"Write", "Edit", "NotebookEdit", "MultiEdit"}
PATH_KEYS = ("file_path", "notebook_path", "path")
PATTERN_KEYS = {"Glob": "pattern", "Grep": "glob"}

def inside(p, base):
    return p == base or p.startswith(base.rstrip(os.sep) + os.sep)

def resolve(p):
    if not isinstance(p, str) or not p or "\0" in p:
        deny("empty or malformed path")
    if p.startswith("~"):
        deny("home-relative path %r is outside the root" % p)
    return os.path.realpath(os.path.join(root, p))

known = tool in WRITE or tool in ("Read", "Grep", "Glob")
present = [k for k in PATH_KEYS if k in inp]
if not known and not present:
    sys.exit(0)

if tool in ("Read",) or tool in WRITE:
    if not present:
        deny("%s without a path" % tool)

out_dir = os.path.join(root, "out")
for k in present:
    v = inp[k]
    if v is None and tool in ("Grep", "Glob"):
        continue
    r = resolve(v)
    if not inside(r, root):
        deny("%s path %r resolves to %s, outside the session root" % (tool, v, r))
    if tool in WRITE:
        od = os.path.realpath(out_dir)
        if not inside(od, root) or r == od or not inside(r, od):
            deny("%s may write only inside out/ (got %s)" % (tool, r))

if tool in WRITE and not present:
    deny("%s without a path" % tool)

pk = PATTERN_KEYS.get(tool)
if pk and inp.get(pk) is not None:
    pat = inp[pk]
    if not isinstance(pat, str) or "\0" in pat:
        deny("malformed %s" % pk)
    parts = pat.replace("\\", "/").split("/")
    if pat.startswith(("/", "~")) or ".." in parts:
        deny("%s %r would leave the session root" % (pk, pat))
sys.exit(0)
'
