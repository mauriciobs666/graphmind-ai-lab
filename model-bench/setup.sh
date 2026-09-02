#!/usr/bin/env bash
set -euo pipefail

# setup.sh — create/refresh the virtualenv for model-bench.
#
# Idempotent: re-running is safe and cheap (pip reports "Requirement already
# satisfied" and exits 0). Ends by importing the package, so a successful run is
# also the component's install smoke test. Mirrors mcp-monitor/setup.sh.
#
# model-bench has NO runtime dependencies by design (see pyproject.toml), so the
# venv exists for the dev extra (pytest, ruff) and for an isolated editable
# install — not to carry a dependency tree.
#
# Usage:
#   model-bench/setup.sh              # create/update model-bench/.venv, install `.[dev]`
#   model-bench/setup.sh --recreate   # delete and rebuild .venv from scratch
#   model-bench/setup.sh --help
#
# The venv lives at model-bench/.venv, dedicated to this component (not shared with
# falkor-chat/server/.venv, cypher-mcp/.venv or mcp-monitor/.venv) — untracked.
#
# Env overrides:
#   PYTHON   interpreter used to create the venv (default: python3; needs >= 3.12)

usage() {
  cat <<'EOF'
Usage: setup.sh [--recreate] [-h|--help]

  (no args)     create model-bench/.venv if missing, install the package with
                its `dev` extra, verify the package imports
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

echo "── installing model-bench[dev] ──────────────────────────"
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$HERE[dev]"

echo "── smoke: importing the package ─────────────────────────"
"$VENV/bin/python" - <<'PY'
import modelbench

print(f"  model-bench {modelbench.__version__}")
PY

cat <<EOF

Setup OK. Next:
  $HERE/.venv/bin/pytest $HERE/tests -q     # run the test suite
  $HERE/.venv/bin/ruff check $HERE          # lint
EOF
