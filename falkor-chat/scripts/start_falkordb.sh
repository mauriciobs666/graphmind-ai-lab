#!/usr/bin/env bash
set -euo pipefail

# Starts FalkorDB in Docker.
# Ports: Redis/FalkorDB on ${FALKORDB_PORT:-6379}, web console on ${FALKORDB_WEB_PORT:-3000}.
# Override defaults with env vars:
#   FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT, FALKORDB_CONTAINER_NAME,
#   FALKORDB_VOLUME (default falkordb-data; a scratch host dir for a dry run, never the live volume),
#   FALKORDB_ENV_FILE (default <falkor-chat>/config/falkordb.env), FALKORDB_THREAD_COUNT,
#   FALKORDB_MODULE_ARGS, FALKORDB_ARGS (a full module-arg string, used as-is; the rollback path)
#
# Module args (GRAPH.CONFIG at load time): composed as
#   "THREAD_COUNT $FALKORDB_THREAD_COUNT $FALKORDB_MODULE_ARGS" from config/falkordb.env, and
# passed as the image's FALKORDB_ARGS, which REPLACES the image's own string (it does not add to
# it). Values and sizing: falkor-chat/docs/plans/graph-config-defaults-graph.md.
#
# Data persistence: graphs are stored in a named Docker volume (falkordb-data)
# mounted at /var/lib/falkordb/data — the image's actual Redis `dir`
# (FALKORDB_DATA_PATH). NOTE: mounting at /data (the image's legacy VOLUME)
# persists NOTHING — live-verified 2026-07-04 on falkordb/falkordb:edge.
# The volume survives container restarts; remove it explicitly with:
#   docker volume rm falkordb-data

usage() {
  cat <<EOF
Usage: start_falkordb.sh [-d|--detach] [-h|--help]

  (no args)        run in the foreground, streaming logs (Ctrl+C to stop)
  -d, --detach     run headless in the background; prints how to stop
  -h, --help       show this help

Env overrides: FALKORDB_IMAGE, FALKORDB_PORT, FALKORDB_WEB_PORT, FALKORDB_CONTAINER_NAME,
               FALKORDB_VOLUME (default falkordb-data),
               FALKORDB_ENV_FILE (default <falkor-chat>/config/falkordb.env),
               FALKORDB_THREAD_COUNT, FALKORDB_MODULE_ARGS (override the env file's values),
               FALKORDB_ARGS (full module-arg string, used as-is; the env file is not read)
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

FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:v4.18.11}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORDB_WEB_PORT="${FALKORDB_WEB_PORT:-3000}"
FALKORDB_CONTAINER_NAME="${FALKORDB_CONTAINER_NAME:-falkordb-dev}"
FALKORDB_VOLUME="${FALKORDB_VOLUME:-falkordb-data}"
FALKORDB_ENV_FILE="${FALKORDB_ENV_FILE:-$(cd "$(dirname "$0")/.." && pwd)/config/falkordb.env}"

# Module args: a full FALKORDB_ARGS in the environment wins as-is (the rollback path). Otherwise
# compose it from the env file, where an already-set FALKORDB_THREAD_COUNT/FALKORDB_MODULE_ARGS wins.
# The file is parsed, not sourced: its values contain spaces. compose.yaml reads the same file under
# dotenv rules (quotes, inline `#`, `$`, trimming, last duplicate wins), so this reader accepts only
# the subset both read identically and refuses everything else: the two keys, once each, as
# KEY=value with a value of [A-Za-z0-9_.-] words separated by single spaces.
if [ -z "${FALKORDB_ARGS:-}" ]; then
  if [ ! -f "$FALKORDB_ENV_FILE" ] || [ ! -r "$FALKORDB_ENV_FILE" ]; then
    echo "start_falkordb.sh: env file '$FALKORDB_ENV_FILE' not found or not readable (set FALKORDB_ENV_FILE or FALKORDB_ARGS)" >&2
    exit 1
  fi
  value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'
  seen=" "
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      FALKORDB_THREAD_COUNT|FALKORDB_MODULE_ARGS) ;;
      *) echo "start_falkordb.sh: unknown key '$key' in $FALKORDB_ENV_FILE" >&2; exit 1 ;;
    esac
    if [ "$line" = "$key" ] || [[ ! "$value" =~ $value_re ]]; then
      echo "start_falkordb.sh: $key in $FALKORDB_ENV_FILE: write KEY=value, the value as words of letters, digits, '_', '.', '-' separated by single spaces (no quotes, comments, '\$' or '=')" >&2
      exit 1
    fi
    case "$seen" in *" $key "*)
      echo "start_falkordb.sh: duplicate key '$key' in $FALKORDB_ENV_FILE" >&2; exit 1 ;;
    esac
    seen="$seen$key "
    if [ -z "${!key+x}" ]; then printf -v "$key" '%s' "$value"; fi
  done < "$FALKORDB_ENV_FILE"
  FALKORDB_ARGS="THREAD_COUNT ${FALKORDB_THREAD_COUNT:?set in $FALKORDB_ENV_FILE} ${FALKORDB_MODULE_ARGS:?set in $FALKORDB_ENV_FILE}"
fi

echo "Starting FalkorDB container '${FALKORDB_CONTAINER_NAME}' (${FALKORDB_IMAGE})..."
echo "Redis/FalkorDB endpoint: localhost:${FALKORDB_PORT}"
echo "Web console:             http://localhost:${FALKORDB_WEB_PORT}"
echo "Data volume:             ${FALKORDB_VOLUME} (persists across restarts)"
echo "Module args:             ${FALKORDB_ARGS}"

if [ "$DETACH" -eq 1 ]; then
  docker run \
    --name "${FALKORDB_CONTAINER_NAME}" \
    -p "${FALKORDB_PORT}:6379" \
    -p "${FALKORDB_WEB_PORT}:3000" \
    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \
    -v "${FALKORDB_VOLUME}":/var/lib/falkordb/data \
    --rm -d "${FALKORDB_IMAGE}"
  echo "Started in the background. Stop with: docker stop ${FALKORDB_CONTAINER_NAME}"
else
  docker run \
    --name "${FALKORDB_CONTAINER_NAME}" \
    -p "${FALKORDB_PORT}:6379" \
    -p "${FALKORDB_WEB_PORT}:3000" \
    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \
    -v "${FALKORDB_VOLUME}":/var/lib/falkordb/data \
    --rm -it "${FALKORDB_IMAGE}"
fi
