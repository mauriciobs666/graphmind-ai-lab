#!/bin/bash
# mutants_guard_paths.sh — mutation check for guard-paths.sh (S52). Each mutant breaks one rule;
# test_guard_paths.sh must FAIL against it. A surviving mutant means a rule has no test.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../guard-paths.sh"
TMP="$(mktemp -d)"; trap 'rm -rf -- "$TMP"' EXIT

# The up-front ".."-component check in rule 3 is now subsumed by the pattern analysis (no test can tell
# them apart), so it has no mutant; the absolute-prefix part is not subsumed for in-root absolute patterns.
# A mutant that makes the guard hang is killed by the 5 s cap inside the tests or by the 120 s `timeout`
# around each mutant run below (exit 124/137 is non-zero, so it counts as killed).
# No mutant for the per-pattern variant budget alone (backed by the brace and wildcard caps).
# name|old|new   (old must occur exactly once in the guard)
mutants=(
'symlinks not resolved|os.path.realpath(os.path.join(root, p))|os.path.abspath(os.path.join(root, p))'
'root not symlink-resolved|root = os.path.realpath(cwd)|root = os.path.abspath(cwd)'
'prefix match without separator|return p == base or p.startswith(base.rstrip(os.sep) + os.sep)|return p == base or p.startswith(base)'
'outside-root check removed|    if not inside(r, root):|    if False:'
'write allowed outside out/|if not inside(od, root) or r == od or not inside(r, od):|if False:'
'write to out/ itself allowed|or r == od or not inside(r, od)|or not inside(r, od)'
'fail-open on bad input|    deny("unreadable hook input (%s)" % exc)|    sys.exit(0)'
'tilde path allowed|    if p.startswith("~"):|    if False:'
'empty path allowed|if not isinstance(p, str) or not p or|if not isinstance(p, str) or False and not p or'
'absolute pattern prefix check dropped (in-root absolute pattern allowed)|    if pat.startswith(("/", "~")) or ".." in parts:|    if ".." in parts:'
'unknown tool with path skipped|if not known and not present:|if not known:'
'read without path allowed|        if not present:\n            deny("%s without a path" % tool)|        pass'
'cwd field ignored (process cwd used)|    cwd = data.get("cwd")|    cwd = None'
'internal errors not caught (exit 1 on surrogate)|except Exception as exc:  # any internal failure|except ZeroDivisionError as exc:  # any internal failure'
'pattern_escapes not called (S54 D1)|        if pattern_escapes(base_dir, pat):|        if False:'
'brace expansion off|            m = re.search(r"\{([^{}]*)\}", cur)|            m = None'
'unescaped (backslash) reading dropped|dict.fromkeys((pat, unescaped))|dict.fromkeys((pat,))'
'reading budget unbounded (all overflow checks off)|    MAXV = 512 |    MAXV = 10**9 '
'as-written reading dropped|dict.fromkeys((pat, unescaped))|dict.fromkeys((unescaped,))'
'brace cap removed (cost)|                if len(out) > MAXV:|                if False:'
'wildcard-product cap removed (cost)|            if n > MAXV:\n                return None|            if False:\n                return None'
'brace overflow safe|        if alts is None:\n            return True|        if alts is None:\n            return False'
'class reading removed|parts.append(["x"] + [ch for ch in SPECIALS if class_may_match(body, ch)])|parts.append(["x"])'
'? reading removed|                parts.append(["x", *SPECIALS])|                parts.append(["x"])'
'dot-dot component not rejected in base|        if ".." in comps:\n            return None|        if False:\n            return None'
'tilde not rejected in base|        if pat.startswith("~"):\n            return None|        if False:\n            return None'
'variant outside root allowed|if b is None or not inside(b, root):|if b is None:'
'pattern base ignores path arg|            base_dir = resolve(inp["path"])|            base_dir = root'
'non-string path accepted|        deny("empty or malformed path")|        return root'
)
survived=0 killed=0
i=0
for m in "${mutants[@]}"; do
	i=$((i+1))
	IFS='|' read -r name old new <<<"$m"
	f="$TMP/mut$i.sh"
	/usr/bin/python3 -I - "$SRC" "$f" "$old" "$new" <<'PY' || { echo "BAD MUTANT SPEC: $name"; exit 3; }
import sys
src, dst, old, new = sys.argv[1:5]
old = old.replace("\\n", "\n"); new = new.replace("\\n", "\n")
t = open(src).read()
if t.count(old) != 1:
    sys.stderr.write("old text occurs %d times\n" % t.count(old)); sys.exit(1)
open(dst, "w").write(t.replace(old, new))
PY
	chmod +x "$f"
	if GUARD="$f" timeout -s KILL 120 "$HERE/test_guard_paths.sh" >/dev/null 2>&1; then
		echo "SURVIVED  $name"; survived=$((survived+1))
	else
		echo "killed    $name"; killed=$((killed+1))
	fi
done

# settings.json mutants: the test takes SETTINGS=<path>
S_SRC="$HERE/../settings.json"
smutants=(
'settings command reverted to exec (guard exit 1 passes through)|; "$CCG_GUARD"; rc=$?; [ "$rc" -eq 0 ] || exit 2|; exec "$CCG_GUARD"'
'settings command ignores guard exit code|; rc=$?; [ "$rc" -eq 0 ] || exit 2|; true'
)
for m in "${smutants[@]}"; do
	i=$((i+1))
	IFS='|' read -r name old new <<<"$m"
	f="$TMP/smut$i.json"
	/usr/bin/python3 -I - "$S_SRC" "$f" "$old" "$new" <<'PY' || { echo "BAD MUTANT SPEC: $name"; exit 3; }
import sys, json
src, dst, old, new = sys.argv[1:5]
s = json.load(open(src)); h = s["hooks"]["PreToolUse"][0]["hooks"][0]
if h["command"].count(old) != 1:
    sys.stderr.write("old text occurs %d times\n" % h["command"].count(old)); sys.exit(1)
h["command"] = h["command"].replace(old, new)
json.dump(s, open(dst, "w"))
PY
	if SETTINGS="$f" timeout -s KILL 120 "$HERE/test_guard_paths.sh" >/dev/null 2>&1; then
		echo "SURVIVED  $name"; survived=$((survived+1))
	else
		echo "killed    $name"; killed=$((killed+1))
	fi
done
echo "baseline:"; "$HERE/test_guard_paths.sh" | tail -1
printf '\n%d killed, %d survived\n' "$killed" "$survived"
[ "$survived" = 0 ]
