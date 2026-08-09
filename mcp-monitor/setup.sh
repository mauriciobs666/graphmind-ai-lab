#!/usr/bin/env bash
set -euo pipefail

# setup.sh — create/refresh the virtualenv for mcp-monitor.
#
# Idempotent: re-running is safe and cheap (pip reports "Requirement already
# satisfied" and exits 0). Ends by importing the runtime dependency and the
# package itself, so a successful run is also the component's dependency
# smoke test. Mirrors cpg/mcp/setup.sh.
#
# Usage:
#   mcp-monitor/setup.sh              # create/update mcp-monitor/.venv, install `.[dev]`
#   mcp-monitor/setup.sh --recreate   # delete and rebuild .venv from scratch
#   mcp-monitor/setup.sh --help
#
# The venv lives at mcp-monitor/.venv, dedicated to this component (not shared
# with falkor-chat/server/.venv or cpg/mcp/.venv) — untracked, the repo-root
# .gitignore already ignores `.venv`.
#
# Env overrides:
#   PYTHON   interpreter used to create the venv (default: python3; needs >= 3.12)

usage() {
  cat <<'EOF'
Usage: setup.sh [--recreate] [-h|--help]

  (no args)     create mcp-monitor/.venv if missing, install the package with
                its `dev` extra, verify the runtime imports
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

echo "── installing mcp-monitor[dev] ──────────────────────────"
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$HERE[dev]"

echo "── smoke: importing runtime dependencies ────────────────"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

import mcp.client.stdio  # noqa: F401 — the stdio transport client.py builds on
import mcp.client.streamable_http  # noqa: F401 — the HTTP transport client.py builds on
import mcp_monitor  # noqa: F401 — the package itself

print(f"  mcp        {version('mcp')}")
print(f"  mcp-monitor {mcp_monitor.__version__}")
PY

cat <<EOF

Setup OK. Next:
  $HERE/.venv/bin/pytest $HERE/tests -q     # run the test suite
  $HERE/run.sh --config config.example.toml # run mcp-monitor
EOF
