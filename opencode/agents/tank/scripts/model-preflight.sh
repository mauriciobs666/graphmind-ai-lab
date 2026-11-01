#!/usr/bin/env bash
# model-preflight.sh [<lmstudio-model-id>] — refuse to start tank unless its
# LM Studio model is loaded as exactly one copy, at the required context and
# parallel (docs/plans/standard-local-model.md §3.3). Called by the outer
# scripts only; exit 0 continues, exit 1 means "do not run opencode".
set -euo pipefail

# FR-7: minimum context of the one loaded copy (README Prerequisites).
TANK_MIN_CONTEXT_LENGTH=20480
# FR-7: parallel slots the one loaded copy must have (README Prerequisites).
TANK_REQUIRED_PARALLEL=1

if [ "${TANK_MODEL_PREFLIGHT:-}" = "0" ]; then
  echo "WARNING: TANK_MODEL_PREFLIGHT=0, model preflight skipped" >&2
  exit 0
fi

TANK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$TANK_DIR/opencode.json"

# Prints "<id> <root>" (config-derived); id is $1 when given.
read -r MODEL_ID ROOT < <(python3 - "$CONFIG" "${1:-}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
mid = sys.argv[2] or cfg["agent"]["tank"]["model"].removeprefix("lmstudio/")
root = cfg["provider"]["lmstudio"]["options"]["baseURL"].rstrip("/").removesuffix("/v1")
print(mid, root)
PY
)

BODY="$(curl -sf --max-time 5 "$ROOT/api/v1/models" 2>/dev/null)" || {
  echo "REFUSED: LM Studio not reachable at $ROOT; start LM Studio's server" >&2
  exit 1
}

python3 - "$MODEL_ID" "$ROOT" "$TANK_MIN_CONTEXT_LENGTH" "$TANK_REQUIRED_PARALLEL" "$BODY" <<'PY'
import json, sys
mid, root, minctx, par, body = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def die(msg):
    print(msg, file=sys.stderr); sys.exit(1)
try:
    models = json.loads(body)["models"]
    assert isinstance(models, list)
except Exception:
    die(f"REFUSED: LM Studio not reachable at {root} (answer is not the /api/v1/models JSON); start LM Studio's server")
entry = next((m for m in models if m.get("key") == mid), None)
if entry is None:
    die(f"REFUSED: {mid} not in LM Studio; check the id in opencode.json")
insts = entry.get("loaded_instances") or []
others = [i["id"] for m in models if m.get("type") == "llm" and m.get("key") != mid
          for i in (m.get("loaded_instances") or [])]
def note_others():
    if others:
        print(f"NOTE: other LLM instance(s) loaded: {', '.join(others)}", file=sys.stderr)
if not insts:
    print(f"NOTE: {mid} not loaded; the first request JIT-loads it, and it must come up at >= {minctx} with parallel {par}. Unless the host has saved per-model defaults for it (qwen has; ministral has not), load it by hand as in README Prerequisites", file=sys.stderr)
    note_others(); sys.exit(0)
ids = [i.get("id") for i in insts]
if len(insts) >= 2 or ids[0] != mid:
    die(f"REFUSED: {len(insts)} copies of {mid} loaded ({', '.join(map(str, ids))}); unload the extra one(s) with `lms unload <id:n>`")
cfg = insts[0].get("config") or {}
ctx, p = cfg.get("context_length"), cfg.get("parallel")
fix = f"when nothing else is using it, `lms unload {mid}` and reload it at >= {minctx} with parallel {par} (README Prerequisites)"
if not isinstance(ctx, int) or ctx < minctx:
    die(f"REFUSED: {mid} loaded at {ctx}, tank needs >= {minctx}; {fix}")
if p != par:
    die(f"REFUSED: {mid} loaded with parallel {p}, tank needs {par} (FR-7); {fix}")
print(f"OK: {mid} loaded, context {ctx}, parallel {p}", file=sys.stderr)
note_others()
PY
