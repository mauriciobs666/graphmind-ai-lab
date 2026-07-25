#!/usr/bin/env bash
set -euo pipefail

# setup.sh — create/refresh the virtualenv for the `cpg` MCP server.
#
# Idempotent: re-running is safe and cheap (pip reports "Requirement already
# satisfied" and exits 0). Ends by importing the two runtime dependencies, so a
# successful run is also the component's dependency smoke test.
#
# Usage:
#   cpg/mcp/setup.sh              # create/update cpg/mcp/.venv, then smoke-test imports
#   cpg/mcp/setup.sh --recreate   # delete and rebuild .venv from scratch
#   cpg/mcp/setup.sh --help
#
# The venv lives at cpg/mcp/.venv, deliberately NOT shared with
# falkor-chat/server/.venv: that one pins a chat application's dependency set,
# and coupling them means a falkor-chat bump can break CPG query access.
# It is untracked — the repo-root .gitignore already ignores `.venv`.
#
# Env overrides:
#   PYTHON   interpreter used to create the venv (default: python3; needs >= 3.12)

usage() {
  cat <<'EOF'
Usage: setup.sh [--recreate] [-h|--help]

  (no args)     create cpg/mcp/.venv if missing, install requirements-dev.txt,
                verify the runtime imports
  --recreate    remove an existing .venv first (use after a Python upgrade or a
                half-installed venv)
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

# Resolve paths from this script's own location so the working directory does
# not matter (this monorepo is normally entered from a component subdirectory).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "setup.sh: '$PYTHON' not found on PATH. Install Python >= 3.12 or set PYTHON=." >&2
  exit 1
fi

# Match falkor-chat/server's requires-python — the mcp/falkordb pins are only
# verified on 3.12+.
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
  # A directory without bin/python is a failed earlier run: clear it rather
  # than letting `venv` fail or silently reuse it.
  if [ -d "$VENV" ]; then rm -rf "$VENV"; fi
  echo "── creating venv with $("$PYTHON" -V 2>&1) ──────────────────────"
  "$PYTHON" -m venv "$VENV"
fi

echo "── installing requirements-dev.txt ──────────────────────"
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$HERE/requirements-dev.txt"

echo "── smoke: importing runtime dependencies ────────────────"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

import falkordb  # noqa: F401  — the FalkorDB client server.py connects with
import mcp.server.fastmcp  # noqa: F401  — the FastMCP surface server.py builds on

for dist in ("mcp", "falkordb"):
    print(f"  {dist:<9} {version(dist)}")
PY

cat <<'EOF'

Setup OK. Next:
  cpg/mcp/.venv/bin/python -c "import mcp.server.fastmcp, falkordb"   # dependency smoke test
  cpg/mcp/.venv/bin/pytest cpg/mcp/tests -q                           # available once S2 lands the server
EOF
