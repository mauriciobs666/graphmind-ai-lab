#!/bin/bash
# test_guard_paths.sh — S52 tests for guard-paths.sh (docs/plans/connected-code-graph-p0-2.md §9.1:
# denies paths outside the root given absolute, relative and symlinked inputs; denies writes
# outside out/; allows in-root reads). Also checks settings.json carries the guard and nothing else.
#   GUARD=<path> overrides the script under test (used by mutants_guard_paths.sh).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${GUARD:-$HERE/../guard-paths.sh}"
SETTINGS="${SETTINGS:-$HERE/../settings.json}"
TMP="$(mktemp -d)"
trap 'rm -rf -- "$TMP"' EXIT

ROOT="$TMP/roles/KA-http"
OUTSIDE="$TMP/outside"
mkdir -p "$ROOT/out" "$ROOT/kit-src/pkg" "$OUTSIDE" "$TMP/roles/KB-http/out"
echo secret >"$OUTSIDE/secret.txt"
echo sib >"$TMP/roles/KB-http/out/KB.jsonl"
echo hi >"$ROOT/kit-src/pkg/app.py"
echo brief >"$ROOT/BRIEF.md"
ln -s "$OUTSIDE" "$ROOT/kit-src/escape-dir"
ln -s "$OUTSIDE/secret.txt" "$ROOT/kit-src/escape-file"
ln -s "$OUTSIDE" "$ROOT/out/escape-out"
ln -s "$ROOT/kit-src/pkg/app.py" "$ROOT/out/inner-link"
ln -s "$ROOT/out" "$ROOT/kit-src/to-out"
ln -s "$ROOT" "$TMP/rootlink"

pass=0 fail=0
ok()  { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }

# call <json-input> -> sets RC
call() { printf '%s' "$1" | env -i PATH=/usr/bin:/bin "$GUARD" >/dev/null 2>&1; RC=$?; }
# mk <tool> <key> <value> -> hook JSON with cwd=$ROOT
mk() { /usr/bin/python3 -I -c 'import json,sys; print(json.dumps({"tool_name":sys.argv[1],"tool_input":{sys.argv[2]:sys.argv[3]},"cwd":sys.argv[4]}))' "$1" "$2" "$3" "$ROOT"; }
allow() { local n="$1"; call "$(mk "${@:2}")"; [ "$RC" = 0 ] && ok "$n" || bad "$n (rc=$RC, want 0)"; }
deny()  { local n="$1"; call "$(mk "${@:2}")"; [ "$RC" = 2 ] && ok "$n" || bad "$n (rc=$RC, want 2)"; }

# 3. allows in-root reads
allow "read absolute in-root"          Read file_path "$ROOT/kit-src/pkg/app.py"
allow "read relative in-root"          Read file_path "kit-src/pkg/app.py"
allow "read dot-dot staying inside"    Read file_path "kit-src/../BRIEF.md"
allow "read own out/ file"             Read file_path "out/KA.jsonl"
allow "grep default path (cwd)"        Grep pattern "foo"
allow "grep path in root"              Grep path "kit-src"
allow "glob path in root"              Glob path "$ROOT/kit-src"
allow "read via link that stays inside" Read file_path "out/inner-link"
# the session cwd itself reached through a symlink: in-root stays allowed, escapes stay denied
call "$(/usr/bin/python3 -I -c 'import json,sys; print(json.dumps({"tool_name":"Read","tool_input":{"file_path":"kit-src/pkg/app.py"},"cwd":sys.argv[1]}))' "$TMP/rootlink")"
[ "$RC" = 0 ] && ok "cwd via symlink: in-root read allowed" || bad "cwd via symlink: in-root read allowed (rc=$RC)"
call "$(/usr/bin/python3 -I -c 'import json,sys; print(json.dumps({"tool_name":"Write","tool_input":{"file_path":"out/KA.jsonl"},"cwd":sys.argv[1]}))' "$TMP/rootlink")"
[ "$RC" = 0 ] && ok "cwd via symlink: out/ write allowed" || bad "cwd via symlink: out/ write allowed (rc=$RC)"
call "$(/usr/bin/python3 -I -c 'import json,sys; print(json.dumps({"tool_name":"Read","tool_input":{"file_path":"../outside/secret.txt"},"cwd":sys.argv[1]}))' "$TMP/rootlink")"
[ "$RC" = 2 ] && ok "cwd via symlink: dot-dot escape denied" || bad "cwd via symlink: dot-dot escape denied (rc=$RC)"
# 1. denies outside the root: absolute, relative, symlinked
deny  "read absolute repo file"        Read file_path "/home/never/repo/AGENTS.md"
deny  "read /etc/passwd"               Read file_path "/etc/passwd"
deny  "read sibling role out/"         Read file_path "$TMP/roles/KB-http/out/KB.jsonl"
deny  "read relative dot-dot"          Read file_path "../../outside/secret.txt"
deny  "read relative triple dot-dot"   Read file_path "../../../../../../etc/passwd"
deny  "read inside-then-out dot-dot"   Read file_path "kit-src/../../outside/secret.txt"
deny  "read symlinked file escaping"   Read file_path "kit-src/escape-file"
deny  "read through symlinked dir"     Read file_path "kit-src/escape-dir/secret.txt"
deny  "read through out/ link escape"  Read file_path "out/escape-out/secret.txt"
deny  "read home-relative ~"           Read file_path "~/.claude/history.jsonl"
deny  "read /tmp"                      Read file_path "/tmp/anything"
deny  "read sibling by prefix name"    Read file_path "$ROOT-evil/x"
deny  "read empty path"                Read file_path ""
deny  "grep path ../../.."             Grep path "../../.."
deny  "grep path absolute outside"     Grep path "$OUTSIDE"
deny  "glob path absolute outside"     Glob path "/etc"
deny  "glob path symlink dir escape"   Glob path "kit-src/escape-dir"
deny  "glob pattern absolute"          Glob pattern "/etc/*"
deny  "glob pattern dot-dot"           Glob pattern "../../*/out/*.jsonl"
deny  "glob pattern ~"                 Glob pattern "~/.claude/*"
deny  "grep glob absolute"             Grep glob "/etc/*"
deny  "grep glob dot-dot"              Grep glob "../*"
allow "glob pattern relative"          Glob pattern "kit-src/**/*.py"
allow "grep glob relative"             Grep glob "*.py"
# 2. denies writes outside out/
allow "write out/ file"                Write file_path "out/KA.jsonl"
allow "write out/ absolute"            Write file_path "$ROOT/out/KA.jsonl"
allow "edit out/ file"                 Edit file_path "out/KA.jsonl"
deny  "write BRIEF.md"                 Write file_path "BRIEF.md"
deny  "write kit-src file"             Write file_path "kit-src/pkg/app.py"
deny  "edit kit-src file"              Edit file_path "kit-src/pkg/app.py"
deny  "write root-level new file"      Write file_path "notes.md"
deny  "write out dir itself"           Write file_path "out"
deny  "write outside root absolute"    Write file_path "$OUTSIDE/x"
deny  "write out/ dot-dot to root"     Write file_path "out/../BRIEF.md"
deny  "write out/ dot-dot to outside"  Write file_path "out/../../outside/x"
deny  "write through out/ link escape" Write file_path "out/escape-out/x"
deny  "write via kit-src link to out"  Write file_path "kit-src/to-out/../BRIEF.md"
deny  "write sibling role out/"        Write file_path "$TMP/roles/KB-http/out/KB.jsonl"
deny  "write home-relative"            Write file_path "~/x"
deny  "notebook edit outside out/"     NotebookEdit notebook_path "kit-src/a.ipynb"
deny  "write without path"             Write content "x"
deny  "read without path"              Read offset "1"

# fail-closed on bad input
for bad_in in '' 'not json' '[]' '{"tool_name":"Read"}' '{"tool_input":{}}' '{"tool_name":"Read","tool_input":{"file_path":"a"},"cwd":""}' '{"tool_name":"Read","tool_input":{"file_path":5},"cwd":"/"}'; do
	call "$bad_in"; [ "$RC" = 2 ] && ok "fail-closed on input: ${bad_in:-<empty>}" || bad "fail-closed on input: ${bad_in:-<empty>} (rc=$RC)"
done
# a tool without any path field is not this guard's concern
call '{"tool_name":"SomethingElse","tool_input":{"x":1},"cwd":"/tmp"}'; [ "$RC" = 0 ] && ok "pathless unknown tool passes" || bad "pathless unknown tool passes (rc=$RC)"
# an unknown tool that carries a path field IS checked
call "$(mk SomethingElse path /etc/passwd)"; [ "$RC" = 2 ] && ok "unknown tool with outside path denied" || bad "unknown tool with outside path denied (rc=$RC)"
# cwd absent: falls back to the process cwd
OUT="$(cd "$ROOT" && printf '%s' '{"tool_name":"Read","tool_input":{"file_path":"/etc/passwd"}}' | env -i PATH=/usr/bin:/bin "$GUARD" 2>&1 >/dev/null)"; RC=$?
[ "$RC" = 2 ] && ok "no cwd field: outside read denied" || bad "no cwd field: outside read denied (rc=$RC)"
(cd "$ROOT" && printf '%s' '{"tool_name":"Read","tool_input":{"file_path":"BRIEF.md"}}' | env -i PATH=/usr/bin:/bin "$GUARD" >/dev/null 2>&1) && ok "no cwd field: inside read allowed" || bad "no cwd field: inside read allowed"
# deny carries a reason on stderr
[ -n "$OUT" ] && ok "deny writes a reason to stderr" || bad "deny writes a reason to stderr"

# settings.json: the PreToolUse guard and nothing else
if /usr/bin/python3 -I - "$SETTINGS" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert list(s) == ["hooks"], list(s)
assert list(s["hooks"]) == ["PreToolUse"], list(s["hooks"])
(entry,) = s["hooks"]["PreToolUse"]
for t in ("Read", "Grep", "Glob", "Write", "Edit"):
    assert t in entry["matcher"].split("|"), t
(h,) = entry["hooks"]
assert h["type"] == "command" and "$CCG_GUARD" in h["command"] and "exit 2" in h["command"] and 'exec "$CCG_GUARD"' not in h["command"]
PY
then ok "settings.json: PreToolUse guard only, fail-closed when CCG_GUARD unset"; else bad "settings.json shape"; fi
# the settings command itself fails closed
out="$(env -i PATH=/usr/bin:/bin sh -c "$(/usr/bin/python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hooks"]["PreToolUse"][0]["hooks"][0]["command"])' "$SETTINGS")" </dev/null 2>&1)"; rc=$?
[ "$rc" = 2 ] && ok "settings command denies when CCG_GUARD is unset" || bad "settings command denies when CCG_GUARD is unset (rc=$rc)"
printf '%s' "$(mk Read file_path "$OUTSIDE/secret.txt")" | env -i PATH=/usr/bin:/bin CCG_GUARD="$GUARD" sh -c "$(/usr/bin/python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hooks"]["PreToolUse"][0]["hooks"][0]["command"])' "$SETTINGS")" >/dev/null 2>&1; rc=$?
[ "$rc" = 2 ] && ok "settings command routes to the guard (deny passes through)" || bad "settings command routes to the guard (rc=$rc)"

# exit-code probe (S52 gate B-1): over tool x key x value class, the guard must exit 0 or 2 and
# never anything else (only exit 2 blocks a PreToolUse hook; exit 1 fails open); and it must
# deny (2) every outside/malformed class wherever the guard examines that key.
PROBE_OUT="$(GUARD="$GUARD" ROOT="$ROOT" /usr/bin/python3 -I - <<'PY'
import json, os, subprocess
guard, root = os.environ["GUARD"], os.environ["ROOT"]
tools = ["Read", "Grep", "Glob", "Write", "Edit", "NotebookEdit", "MultiEdit", "SomethingElse"]
keys = ["file_path", "notebook_path", "path", "pattern", "glob"]
WRITE = {"Write", "Edit", "NotebookEdit", "MultiEdit"}
PATHK = {"file_path", "notebook_path", "path"}
vals = {  # class -> (value, denied-when-examined)
    "in-root": ("out/KA.jsonl", False),
    "abs-outside": ("/etc/passwd", True),
    "rel-outside": ("../../../../../../etc/passwd", True),
    "symlink-out": ("kit-src/escape-dir/secret.txt", None),  # a pattern is not resolved
    "lone-surrogate": ("a\ud800b", None),
    "surrogate-dotdot": ("\ud800/../../../../../../etc/passwd", True),
    "nul": ("a\0b", True),
    "non-str-int": (5, True),
    "non-str-list": (["/etc/passwd"], True),
    "non-str-dict": ({"a": 1}, True),
    "empty": ("", None),
    "deep-5000": ("a/" * 5000 + "x", None),
    "tilde": ("~/x", True),
}
def run(inp):
    return subprocess.run([guard], input=json.dumps(inp).encode(), capture_output=True,
                          env={"PATH": "/usr/bin:/bin"}, timeout=60).returncode
bad = []; cells = 0
for tool in tools:
    for key in keys:
        for cls, (v, must_deny) in vals.items():
            cells += 1
            rc = run({"tool_name": tool, "tool_input": {key: v}, "cwd": root})
            examined = key in PATHK or (tool == "Glob" and key == "pattern") or (tool == "Grep" and key == "glob")
            if rc not in (0, 2):
                bad.append("rc=%s %s %s %s" % (rc, tool, key, cls))
            elif examined and must_deny and rc != 2:
                bad.append("allowed %s %s %s" % (tool, key, cls))
            elif examined and key in PATHK and cls == "in-root" and rc != 0:
                bad.append("denied in-root %s %s" % (tool, key))
# cwd classes
for name, cwd in [("lone-surrogate", "a\ud800b"), ("nul", "a\0b"), ("int", 5), ("empty", ""), ("missing-dir", "/nonexistent/zz")]:
    cells += 1
    rc = run({"tool_name": "Read", "tool_input": {"file_path": "x"}, "cwd": cwd})
    if rc not in (0, 2): bad.append("rc=%s cwd %s" % (rc, name))
    if name != "missing-dir" and rc != 2: bad.append("cwd %s not denied (rc=%s)" % (name, rc))
print("CELLS", cells)
print("BAD", len(bad))
for b in bad[:15]: print("  " + b)
PY
)"
echo "$PROBE_OUT" | sed 's/^/     /'
[ "$(echo "$PROBE_OUT" | sed -n 's/^BAD //p')" = 0 ] && ok "exit-code probe: only 0/2, outside and malformed denied" || bad "exit-code probe (see above)"

# settings command: any guard exit other than 0 must become exit 2
SETCMD="$(/usr/bin/python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hooks"]["PreToolUse"][0]["hooks"][0]["command"])' "$SETTINGS")"
for stub_rc in 1 3 126 139; do
	printf '#!/bin/sh\nexit %s\n' "$stub_rc" >"$TMP/stub.sh"; chmod +x "$TMP/stub.sh"
	env -i PATH=/usr/bin:/bin CCG_GUARD="$TMP/stub.sh" sh -c "$SETCMD" </dev/null >/dev/null 2>&1; rc=$?
	[ "$rc" = 2 ] && ok "settings command: guard exit $stub_rc becomes 2" || bad "settings command: guard exit $stub_rc gave $rc"
done
printf '#!/bin/sh\nexit 0\n' >"$TMP/stub.sh"
env -i PATH=/usr/bin:/bin CCG_GUARD="$TMP/stub.sh" sh -c "$SETCMD" </dev/null >/dev/null 2>&1; rc=$?
[ "$rc" = 0 ] && ok "settings command: guard exit 0 stays 0" || bad "settings command: guard exit 0 gave $rc"
printf '{"tool_name":"Read","tool_input":{"file_path":"\\ud800/../../../etc/passwd"},"cwd":"%s"}' "$ROOT" | env -i PATH=/usr/bin:/bin CCG_GUARD="$GUARD" sh -c "$SETCMD" >/dev/null 2>&1; rc=$?
[ "$rc" = 2 ] && ok "settings command + guard: lone-surrogate path denied (2)" || bad "settings command + guard: lone-surrogate gave $rc"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]
