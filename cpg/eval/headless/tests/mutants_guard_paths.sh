#!/bin/bash
# mutants_guard_paths.sh — mutation check for guard-paths.sh (S52). Each mutant breaks one rule;
# test_guard_paths.sh must FAIL against it. A surviving mutant means a rule has no test.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../guard-paths.sh"
TMP="$(mktemp -d)"; trap 'rm -rf -- "$TMP"' EXIT

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
'glob pattern check removed|    if pat.startswith(("/", "~")) or ".." in parts:|    if False:'
'dot-dot pattern component allowed|    if pat.startswith(("/", "~")) or ".." in parts:|    if pat.startswith(("/", "~")):'
'unknown tool with path skipped|if not known and not present:|if not known:'
'read without path allowed|    if not present:\n        deny("%s without a path" % tool)|    pass'
'cwd field ignored (process cwd used)|    cwd = data.get("cwd")|    cwd = None'
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
	if GUARD="$f" "$HERE/test_guard_paths.sh" >/dev/null 2>&1; then
		echo "SURVIVED  $name"; survived=$((survived+1))
	else
		echo "killed    $name"; killed=$((killed+1))
	fi
done
echo "baseline:"; "$HERE/test_guard_paths.sh" | tail -1
printf '\n%d killed, %d survived\n' "$killed" "$survived"
[ "$survived" = 0 ]
