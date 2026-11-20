#!/usr/bin/env bash
set -euo pipefail

# setup.sh — create/refresh the virtualenv for coordination-graph.
#
# Idempotent: re-running is safe and cheap. Ends by importing the one runtime
# dependency, so a successful run is also the dependency smoke test.
#
# Usage:
#   coordination-graph/setup.sh              # create/update coordination-graph/.venv
#   coordination-graph/setup.sh --recreate   # delete and rebuild .venv from scratch
#   coordination-graph/setup.sh --help
#
# The venv is this component's own (untracked; .gitignore) — not shared with
# cypher-mcp or falkor-chat, so a bump there cannot break the mirror.
#
# Env overrides:
#   PYTHON   interpreter used to create the venv (default: python3; needs >= 3.12)

usage() {
  cat <<'EOF'
Usage: setup.sh [--recreate] [-h|--help]

  (no args)     create coordination-graph/.venv if missing, install the package
                with its dev extras, verify the runtime import
  --recreate    remove an existing .venv first
  -h, --help    show this help

Env overrides: PYTHON (default: python3)
EOF
}

RECREATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --recreate) RECREATE=1 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "setup.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "setup.sh: '$PYTHON' not found on PATH. Install Python >= 3.12 or set PYTHON=." >&2
  exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "setup.sh: Python >= 3.12 required, found $("$PYTHON" -V 2>&1)." >&2
  exit 1
fi

if [ "$RECREATE" -eq 1 ] && [ -d "$VENV" ]; then
  echo "── removing existing venv ───────────────────────────────"
  rm -rf "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
  echo "── venv present ($("$VENV/bin/python" -V 2>&1)) ─────────────────"
else
  # A directory without bin/python is a failed earlier run: clear it.
  if [ -d "$VENV" ]; then rm -rf "$VENV"; fi
  echo "── creating venv with $("$PYTHON" -V 2>&1) ──────────────────────"
  "$PYTHON" -m venv "$VENV"
fi

echo "── installing coordination-graph[dev] (editable) ────────"
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$HERE[dev]"

echo "── smoke: importing runtime dependencies ────────────────"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

import falkordb  # noqa: F401  — the FalkorDB client the store will use

print(f"  falkordb  {version('falkordb')}")
PY

cat <<'EOF'

Setup OK. Next:
  cd coordination-graph && .venv/bin/python -m pytest -q     # offline parser tests
  coordination-graph/coordgraph.sh --help                     # CLI (lands with later steps)
EOF
