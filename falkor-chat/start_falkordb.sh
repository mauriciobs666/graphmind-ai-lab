#!/usr/bin/env bash
set -euo pipefail

# Starts FalkorDB in Docker (foreground — Ctrl+C to stop).
# Ports: Redis/FalkorDB on ${FALKORDB_PORT:-6379}, web console on ${FALKORDB_WEB_PORT:-3000}.
# Override defaults with env vars:
#   FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT, FALKORDB_CONTAINER_NAME
#
# Data persistence: graphs are stored in a named Docker volume (falkordb-data).
# The volume survives container restarts; remove it explicitly with:
#   docker volume rm falkordb-data

FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:edge}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORDB_WEB_PORT="${FALKORDB_WEB_PORT:-3000}"
FALKORDB_CONTAINER_NAME="${FALKORDB_CONTAINER_NAME:-falkordb-dev}"

echo "Starting FalkorDB container '${FALKORDB_CONTAINER_NAME}' (${FALKORDB_IMAGE})..."
echo "Redis/FalkorDB endpoint: localhost:${FALKORDB_PORT}"
echo "Web console:             http://localhost:${FALKORDB_WEB_PORT}"
echo "Data volume:             falkordb-data (persists across restarts)"

docker run \
  --name "${FALKORDB_CONTAINER_NAME}" \
  -p "${FALKORDB_PORT}:6379" \
  -p "${FALKORDB_WEB_PORT}:3000" \
  -v falkordb-data:/data \
  --rm -it "${FALKORDB_IMAGE}"
