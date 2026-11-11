#!/usr/bin/env bash
# A reduced bootstrap_schema.sh: a wrapper function whose parameters carry the literals of its call sites.
set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"

gquery() {
  # gquery <graph> <cypher>
  redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$1" "$2"
}

gconstraint() {
  redis-cli -h "$HOST" -p "$PORT" GRAPH.CONSTRAINT CREATE "$@"
}

bootstrap_reference() {
  local g="reference"
  gquery "$g" "CREATE INDEX FOR (n:WorkflowDef) ON (n.key)"
  gquery "$g" "CREATE INDEX FOR (n:Entity) ON (n.entityId)"
  gconstraint "$g" UNIQUE NODE Entity PROPERTIES 1 entityId
}

bootstrap_reference
gquery "ws:$1" "CREATE INDEX FOR (n:Message) ON (n.id)"
