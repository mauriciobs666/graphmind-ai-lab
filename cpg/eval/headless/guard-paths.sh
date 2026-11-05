#!/bin/sh
# guard-paths.sh — PreToolUse path guard for the headless ground-truth writers (S52, P0-43).
# docs/plans/connected-code-graph-p0-2.md §3 S52, §6.3.
#
# Contract (Claude Code hook): the hook JSON arrives on stdin (tool_name, tool_input, cwd).
#   exit 0                  allow (no output)
#   exit 2 + stderr reason  deny; the reason goes back to the model
# Fail-closed: unreadable or malformed input, a missing path on a path-bearing tool, or any
# internal error denies (exit 2; exit 1 would NOT block the tool). Only exit 0 allows.
#
# Rules (root = the session's cwd, symlinks resolved):
#   1. Every path a file tool names (file_path, notebook_path, path), made absolute against the
#      root and fully resolved (".." and symlinks), must be the root or inside it.
#   2. A write (Write, Edit, NotebookEdit) must resolve strictly inside <root>/out/.
#   3. A Glob pattern or Grep glob is denied if it is absolute, starts with "~", has a ".."
#      component, or under any reading can name something outside the root: braces expanded,
#      `[class]` and `?` read as "/", "~" or ".", and `\x` read as `x` (same logic and 512-reading
#      shared budget as cpg/eval/tools/audit.py pattern_escapes; overflow denies). Known limit:
#      an honest glob with many `?`/classes is voided (fail closed).
#   4. A tool this guard does not know and that carries no path field is not its concern (the
#      permission layer owns tool availability); one that carries a path field is checked.
# Requires /usr/bin/python3 (the launcher's PATH is /usr/bin:/bin).
PY=/usr/bin/python3
[ -x "$PY" ] || PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || { echo "guard-paths: python3 not found; denying" >&2; exit 2; }
exec "$PY" -I -c '
import itertools, json, os, re, sys

def deny(msg):
    try:
        sys.stderr.buffer.write(("guard-paths: " + msg + "\n").encode("utf-8", "backslashreplace"))
        sys.stderr.buffer.flush()
    except Exception:
        pass
    sys.exit(2)

def main():
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

    # --- pattern analysis: mirrors cpg/eval/tools/audit.py (pattern_escapes and helpers) ---
    MAXV = 512                      # one shared budget of readings per pattern; overflow = escape
    SPECIALS = ("/", "~", ".")

    def expand_braces(pat):
        out, stack = [], [pat]
        while stack:
            cur = stack.pop()
            m = re.search(r"\{([^{}]*)\}", cur)
            if not m:
                out.append(cur)
                if len(out) > MAXV:
                    return None
                continue
            stack.extend(cur[:m.start()] + a + cur[m.end():] for a in m.group(1).split(","))
        return out

    def class_may_match(body, ch):
        if body[:1] in ("!", "^"):
            return True
        chars, i = set(), 0
        while i < len(body):
            if i + 2 < len(body) and body[i + 1] == "-":
                if body[i] <= ch <= body[i + 2]:
                    return True
                i += 3
            else:
                chars.add(body[i])
                i += 1
        return ch in chars

    def wildcard_variants(pat):
        parts, i = [], 0
        while i < len(pat):
            c = pat[i]
            j = -1
            if c == "[":
                k = i + 1 + (1 if pat[i + 1:i + 2] in ("!", "^") else 0)
                k += 1 if pat[k:k + 1] == "]" else 0
                j = pat.find("]", k)
            if c == "[" and j != -1:
                body = pat[i + 1:j]
                parts.append(["x"] + [ch for ch in SPECIALS if class_may_match(body, ch)])
                i = j + 1
            elif c == "?":
                parts.append(["x", *SPECIALS])
                i += 1
            else:
                parts.append([c])
                i += 1
        n = 1
        for p in parts:
            n *= len(p)
            if n > MAXV:
                return None
        return ["".join(v) for v in itertools.product(*parts)]

    def literal_base(base_dir, pat):
        if pat.startswith("~"):
            return None
        comps = pat.replace("\\", "/").split("/")
        if ".." in comps:
            return None
        lit = []
        for c in comps:
            if any(ch in c for ch in "*?[{"):
                break
            lit.append(c)
        prefix = "/".join(lit) if lit else "."
        if pat.startswith("/"):
            prefix = "/" + prefix.lstrip("/") if lit else "/"
        return os.path.realpath(os.path.join(base_dir, prefix))

    def escapes_one(base_dir, pat):
        budget = MAXV
        alts = expand_braces(pat)
        if alts is None:
            return True
        for alt in alts:
            variants = wildcard_variants(alt)
            if variants is None:
                return True
            budget -= len(variants)
            if budget < 0:
                return True
            for v in variants:
                b = literal_base(base_dir, v)
                if b is None or not inside(b, root):
                    return True
        return False

    def pattern_escapes(base_dir, pat):
        # read as written, and with "\x" read as "x" (a backslash may escape a dot)
        unescaped = re.sub(r"\\(.)", r"\1", pat, flags=re.DOTALL)
        return any(escapes_one(base_dir, r) for r in dict.fromkeys((pat, unescaped)))

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
        base_dir = root
        if inp.get("path") is not None:
            base_dir = resolve(inp["path"])   # already checked inside the root above
        if pattern_escapes(base_dir, pat):
            deny("%s %r could leave the session root under some reading (braces, classes, ?, backslash)" % (pk, pat))
    sys.exit(0)


try:
    main()
except Exception as exc:  # any internal failure (e.g. an unencodable path) must deny, never exit 1
    deny("internal error (%s: %s); denying" % (type(exc).__name__, exc))
'
