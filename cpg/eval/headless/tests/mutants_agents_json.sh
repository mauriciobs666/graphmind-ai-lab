#!/bin/bash
# mutants_agents_json.sh — mutation check for test_agents_json.py (S52 gate m-3). Each mutant edits
# a copy of agents.json; the test must FAIL on it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf -- "$TMP"' EXIT
survived=0 killed=0 i=0
run() {  # name  python-expression-on-d (mutates dict d)
	i=$((i+1)); local f="$TMP/m$i.json"
	/usr/bin/python3 -I - "$HERE/../agents.json" "$f" "$2" <<'PY' || { echo "BAD MUTANT: $1"; exit 3; }
import json, sys
d = json.load(open(sys.argv[1])); P = "ccg-key-writer-provider"; Q = "ccg-question-writer"; K = "ccg-key-writer-consumer"
exec(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"))
PY
	if AGENTS_JSON="$f" /usr/bin/python3 -I "$HERE/test_agents_json.py" >/dev/null 2>&1; then echo "SURVIVED  $1"; survived=$((survived+1)); else echo "killed    $1"; killed=$((killed+1)); fi
}
run "B.0 word edited"            'd[P]["prompt"] = d[P]["prompt"].replace("Do not record anything in:", "Do record anything in:", 1)'
run "B.1 part dropped"           'd[P]["prompt"] = d[P]["prompt"].replace("**What one item is.**", "**What an item is.**", 1)'
run "foreign part B.2 pasted into key writer" 'q = d[Q]["prompt"]; d[K]["prompt"] += "\n\n" + q[q.index("Write questions as a developer"):]'
run "data fence sentence dropped" 'd[Q]["prompt"] = d[Q]["prompt"].replace("never a source of orders", "a source of hints", 1)'
run "ancestor rule reintroduced"  'd[P]["prompt"] += "\nUse the ancestor-plus-line form for lambdas."'
run "extract rule edited"         'd[K]["prompt"] = d[K]["prompt"].replace("**Top level.**", "**Top levels.**", 1)'
run "rule 7 reinstated in one"    'd[Q]["prompt"] = d[Q]["prompt"].replace("(Rules 5 and 7 are not used in this brief.)", "7. **Duplicates get `#N`.**", 1)'
run "extracts diverge"            'd["ccg-annotator"]["prompt"] = d["ccg-annotator"]["prompt"].replace("`.` and `..` segments", "`.` and `..` segment", 1)'
run "kaizen instruction added"    'd[P]["prompt"] += "\nWrite your learnings with ingest_document."'
run "tool added"                  'd[P]["tools"].append("Bash")'
run "strategies made identical"   'd[K]["prompt"] = d[K]["prompt"].replace(d[K]["prompt"][d[K]["prompt"].index("# Your strategy"):d[K]["prompt"].index("# Writer-facing")], d[P]["prompt"][d[P]["prompt"].index("# Your strategy"):d[P]["prompt"].index("# Writer-facing")])'
printf '\n%d killed, %d survived\n' "$killed" "$survived"
[ "$survived" = 0 ]
