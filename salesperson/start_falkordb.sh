#!/usr/bin/env bash
set -euo pipefail

# Starts FalkorDB in Docker.
# Ports: Redis/FalkorDB on ${FALKORDB_PORT:-6379}, web console on ${FALKORDB_WEB_PORT:-3000}.
# Override defaults with FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT if needed.

FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:edge}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORDB_WEB_PORT="${FALKORDB_WEB_PORT:-3000}"

echo "Starting FalkorDB container (${FALKORDB_IMAGE})..."
echo "Redis/FalkorDB endpoint: localhost:${FALKORDB_PORT}"
echo "Web console: http://localhost:${FALKORDB_WEB_PORT}"

docker run \
  -p "${FALKORDB_PORT}:6379" \
  -p "${FALKORDB_WEB_PORT}:3000" \
  -it --rm "${FALKORDB_IMAGE}"
