#!/usr/bin/env bash
set -euo pipefail

# Starts FalkorDB in Docker.
# Ports: Redis/FalkorDB on ${FALKORDB_PORT:-6379}, web console on ${FALKORDB_WEB_PORT:-3000}.
# Override defaults with env vars:
#   FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT, FALKORDB_CONTAINER_NAME
#
# Data persistence: graphs are stored in a named Docker volume (falkordb-data).
# The volume survives container restarts; remove it explicitly with:
#   docker volume rm falkordb-data

usage() {
  cat <<EOF
Usage: start_falkordb.sh [-d|--detach] [-h|--help]

  (no args)        run in the foreground, streaming logs (Ctrl+C to stop)
  -d, --detach     run headless in the background; prints how to stop
  -h, --help       show this help

Env overrides: FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT, FALKORDB_CONTAINER_NAME
EOF
}

DETACH=0
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--detach) DETACH=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "start_falkordb.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:edge}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORDB_WEB_PORT="${FALKORDB_WEB_PORT:-3000}"
FALKORDB_CONTAINER_NAME="${FALKORDB_CONTAINER_NAME:-falkordb-dev}"

echo "Starting FalkorDB container '${FALKORDB_CONTAINER_NAME}' (${FALKORDB_IMAGE})..."
echo "Redis/FalkorDB endpoint: localhost:${FALKORDB_PORT}"
echo "Web console:             http://localhost:${FALKORDB_WEB_PORT}"
echo "Data volume:             falkordb-data (persists across restarts)"

if [ "$DETACH" -eq 1 ]; then
  docker run \
    --name "${FALKORDB_CONTAINER_NAME}" \
    -p "${FALKORDB_PORT}:6379" \
    -p "${FALKORDB_WEB_PORT}:3000" \
    -v falkordb-data:/data \
    --rm -d "${FALKORDB_IMAGE}"
  echo "Started in the background. Stop with: docker stop ${FALKORDB_CONTAINER_NAME}"
else
  docker run \
    --name "${FALKORDB_CONTAINER_NAME}" \
    -p "${FALKORDB_PORT}:6379" \
    -p "${FALKORDB_WEB_PORT}:3000" \
    -v falkordb-data:/data \
    --rm -it "${FALKORDB_IMAGE}"
fi
