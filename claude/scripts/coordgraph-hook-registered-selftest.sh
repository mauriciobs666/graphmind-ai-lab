#!/usr/bin/env bash
# Selftest for coordgraph-hook-registered.py (audit-team.sh check 15). Never touches .claude/settings*.json:
# every case is materialised in a temp root.
#  1. Positive control: the EXACT json block of coordination-graph/docs/hook-settings.md must pass, and
#     broken variants of it must fail.
#  2. The golden table coordination-graph/tests/fixtures/hook-settings-cases.json (the hookreg authority's
#     own table): every case's settings / settings_local / script state is built in a temp root and the
#     checker must agree: expected `registered` -> rc 0, anything else -> rc 1. Kill-switch cases (`disabled`)
#     are an enablement state the checker deliberately does not read (the audit NOTEs it), so they must give
#     the settings-only verdict: rc 0 iff the case's settings register the hook.
# Optional $1 = path of a checker copy to test (mutation runs); default: the real one.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H="${1:-$ROOT/claude/scripts/coordgraph-hook-registered.py}"
DOC="$ROOT/coordination-graph/docs/hook-settings.md"
TABLE="$ROOT/coordination-graph/tests/fixtures/hook-settings-cases.json"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
exec python3 - "$H" "$DOC" "$TABLE" "$T" <<'PY'
import copy, json, os, re, stat, subprocess, sys
h, doc, table, t = sys.argv[1:5]
SCRIPT = "coordination-graph/hooks/post-tool-sync.sh"
bad = 0

def build(root, settings, local=None, script="executable"):
    os.makedirs(f"{root}/.claude", exist_ok=True)
    p = f"{root}/{SCRIPT}"
    if script == "directory":
        os.makedirs(p)
    elif script != "missing":
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("#!/usr/bin/env bash\n")
        os.chmod(p, 0o755 if script == "executable" else 0o644)
    def put(name, c):
        d = c if isinstance(c, str) else json.dumps(fill(c, root))
        open(f"{root}/.claude/{name}", "w").write(d)
    put("settings.json", settings)
    if local is not None:
        put("settings.local.json", local)

def fill(v, root):
    if isinstance(v, str): return v.replace("{ROOT}", root)
    if isinstance(v, list): return [fill(x, root) for x in v]
    if isinstance(v, dict): return {k: fill(x, root) for k, x in v.items()}
    return v

def run(root):
    r = subprocess.run([sys.executable, h, root], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip().replace("\n", ";")

def check(name, want, root):
    global bad
    rc, out = run(root)
    if rc == want:
        print(f"ok    {name:44} rc={rc}  {out[:90]}")
    else:
        print(f"FAIL  {name:44} rc={rc} want {want}  {out}"); bad += 1

# 1. positive control + variants (the documented block, verbatim)
block = json.loads(re.search(r"```json\n(.*?)\n```", open(doc).read(), re.S).group(1))
variants = {"doc-exact": (block, 0)}
for ev in ("PostToolUse", "PostToolUseFailure"):
    b = copy.deepcopy(block); del b["hooks"][ev]; variants["doc-without-" + ev] = (b, 1)
b = copy.deepcopy(block); b["hooks"]["PostToolUse"][0]["hooks"][0]["command"] = "/x/other.sh"; variants["doc-wrong-path"] = (b, 1)
b = copy.deepcopy(block); b["hooks"]["PostToolUseFailure"][0]["matcher"] = "Read"; variants["doc-wrong-matcher"] = (b, 1)
b = copy.deepcopy(block)
for ev in b["hooks"]: b["hooks"][ev][0]["matcher"] = "Write|Edit"
variants["doc-no-bash"] = (b, 1)
variants["doc-empty-hooks"] = ({"hooks": {}}, 1)
for name, (s, want) in variants.items():
    r = f"{t}/v-{name}"; build(r, s); check(name, want, r)

# 2. the golden table
cases = json.load(open(table))
for c in cases:
    r = f"{t}/c-{c['name']}"
    build(r, c["settings"], c.get("settings_local"), c.get("script", "executable"))
    if c["expected"] == "disabled":
        want = 0 if isinstance(c["settings"], dict) and c["settings"].get("hooks") else 1
    else:
        want = 0 if c["expected"] == "registered" else 1
    check("table:" + c["name"], want, r)
print(f"\n{len(cases)} table cases + {len(variants)} documented-block variants; {'FAILURES' if bad else 'all agree'}")
sys.exit(1 if bad else 0)
PY
