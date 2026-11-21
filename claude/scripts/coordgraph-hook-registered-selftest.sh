#!/usr/bin/env bash
# Selftest for coordgraph-hook-registered.py (audit-team.sh check 15). Builds settings fixtures in a
# temp dir (never touches .claude/settings*.json). Positive control: the EXACT json block of
# coordination-graph/docs/hook-settings.md must pass; every broken variant must fail by name.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H="$ROOT/claude/scripts/coordgraph-hook-registered.py"
DOC="$ROOT/coordination-graph/docs/hook-settings.md"
NEEDLE="coordination-graph/hooks/post-tool-sync.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
bad=0
python3 - "$DOC" "$T" <<'PY'
import json, re, sys, copy
doc, t = sys.argv[1], sys.argv[2]
block = json.loads(re.search(r"```json\n(.*?)\n```", open(doc).read(), re.S).group(1))
w = lambda n, o: json.dump(o, open(f"{t}/{n}.json", "w"))
w("exact", block)
for ev in ("PostToolUse", "PostToolUseFailure"):
    b = copy.deepcopy(block); del b["hooks"][ev]; w("only_" + ("failure" if ev == "PostToolUse" else "use"), b)
b = copy.deepcopy(block); b["hooks"]["PostToolUse"][0]["hooks"][0]["command"] = "/x/other.sh"; w("wrong_path", b)
b = copy.deepcopy(block); b["hooks"]["PostToolUseFailure"][0]["matcher"] = "Read"; w("wrong_matcher", b)
b = copy.deepcopy(block)
for ev in b["hooks"]: b["hooks"][ev][0]["matcher"] = "Write|Edit"
w("no_bash", b)
w("none", {"hooks": {}})
u = copy.deepcopy(block); del u["hooks"]["PostToolUseFailure"]; w("split_a", u)
l = copy.deepcopy(block); del l["hooks"]["PostToolUse"]; w("split_b", l)
open(f"{t}/garbage.json", "w").write("{not json")
PY
check() { # name expect(0|1) files...
  local name="$1" want="$2"; shift 2
  out="$(python3 "$H" "$NEEDLE" "$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want" ]; then printf 'ok    %-14s rc=%s  %s\n' "$name" "$rc" "$(printf '%s' "$out" | tr '\n' ';')"
  else printf 'FAIL  %-14s rc=%s want %s  %s\n' "$name" "$rc" "$want" "$out"; bad=1; fi
}
check exact 0 "$T/exact.json"
check split_files 0 "$T/split_a.json" "$T/split_b.json"
check only_use 1 "$T/only_use.json"
check only_failure 1 "$T/only_failure.json"
check wrong_path 1 "$T/wrong_path.json"
check wrong_matcher 1 "$T/wrong_matcher.json"
check no_bash 1 "$T/no_bash.json"
check none 1 "$T/none.json"
check garbage 1 "$T/garbage.json" "$T/absent.json"
exit "$bad"
