#!/usr/bin/env bash
set -euo pipefail

# run.sh — launch the model-bench CLI from the component venv.
#
# Usage (from S1 onward): model-bench/run.sh <command> [options]
#
# Everything is resolved from this script's own location, so the working
# directory the caller happens to be in does not matter.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "model-bench/run.sh: no venv at $HERE/.venv — run model-bench/setup.sh first." >&2
  exit 1
fi

# --- S0 guard -------------------------------------------------------------
# There is no CLI yet: stage S0 of docs/plans/small-model-benchmarking.md builds
# only the component skeleton, and `modelbench/__main__.py` + `modelbench/cli.py`
# arrive in S1. Failing loudly with a named reason beats `exec`-ing into a bare
# "No module named modelbench.__main__" traceback, and beats pretending to run.
# S1 deletes this block; the line below it is then the whole body of the script.
if ! "$PY" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("modelbench.__main__") else 1)'; then
  cat >&2 <<'EOF'
model-bench/run.sh: nothing to launch yet.

The component skeleton is in place (stage S0), but the CLI — `modelbench/cli.py`
and `modelbench/__main__.py` — is stage S1 of docs/plans/small-model-benchmarking.md.
Until it lands, the only things that run here are:

  model-bench/setup.sh                      # create/refresh the venv
  model-bench/.venv/bin/pytest -q           # the (currently empty) suite
  model-bench/.venv/bin/ruff check model-bench
EOF
  exit 1
fi
# --- end S0 guard ---------------------------------------------------------

exec "$PY" -m modelbench "$@"
