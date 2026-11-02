#!/usr/bin/env bash
# Run the job's HTTP surface for local development.
set -euo pipefail
exec uvicorn job.trigger_server:app --host 0.0.0.0 --port 8000
