#!/usr/bin/env bash
# seed_agent_team.sh — register every claude/ subagent as an Agent node in the
# ws:agent-team workspace (claude/docs/plans/agent-knowledge-base-strategy.md
# §4.2, Track 1 Stage 2), so `ingest_document`'s optional `produced_by`
# (§4.1, K-030) has a real `Agent` to resolve against for each of them.
#
# Usage:
#   ./scripts/seed_agent_team.sh [<workspaceId>]   # default: $FALKORCHAT_WS_ID or "agent-team"
#
# Run after ./scripts/bootstrap_schema.sh agent-team (needs the Agent.agentId
# index + uniqueness constraint that creates — the same one every other
# workspace's Agent node already relies on).
#
# What it seeds (idempotent — safe to re-run):
#   * Agent {agentId}   — one per claude/AGENTS.md's "Agents" roster, guarded
#                          ensure (mirrors seed_demo.sh's own Agent block byte-
#                          for-byte: an id already held by a User is refused,
#                          not silently skipped or overwritten — same
#                          cross-label collision guard `ensure_agent`/
#                          `ensure_user` enforce in the repository layer).
#
# The roster below is a **point-in-time copy** of claude/AGENTS.md's "Agents"
# section (verified 2026-09-18) — it does NOT read that file at run time.
# claude/AGENTS.md's own "Maintenance rules" section already requires touching
# "the name roster" whenever an agent is added/renamed/removed; whoever next
# edits that rule to enumerate every touch point should add this script to it
# (agent-knowledge-base-strategy.md §4.2 names this follow-up explicitly —
# not wired here, out of this script's own scope).
#
# `name` is set equal to `agentId` — this roster has no separate display-name
# convention, and `name` plays no role in `produced_by` resolution (only
# `agentId` is matched against) — a stand-in, not a claim that it's the ideal
# display name.
#
# Env vars (all optional):
#   FALKORDB_HOST      (default: 127.0.0.1)
#   FALKORDB_PORT      (default: 6379)
#   FALKORCHAT_WS_ID   (default: agent-team)   — workspace id (graph key ws:<id>)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-agent-team}}"

G="ws:${WS_ID}"

# claude/AGENTS.md "Agents" section roster, 2026-09-18.
AGENTS=(
  teco tico architect coder tdd-engineer frontend-engineer qa-engineer
  analyst data-scientist graph-dba devops security-expert cobb
)

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}

echo "── seeding ${#AGENTS[@]} agents into ${G} ─────────────────────────────"

n=0
for agent_id in "${AGENTS[@]}"; do
  n=$((n + 1))
  echo "[${n}/${#AGENTS[@]}] Agent ${agent_id}"
  # Guarded ensure — byte-for-byte the same shape as seed_demo.sh's Agent
  # block (an id already held by a User is refused, never silently skipped
  # or overwritten; created=false, existed=false would mean corruption —
  # this script does not create any User, so that branch cannot fire here).
  redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
    "CYPHER agentId='${agent_id}' name='${agent_id}'
     OPTIONAL MATCH (a:Agent {agentId: \$agentId})
     OPTIONAL MATCH (u:User {userId: \$agentId})
     WITH a, u, (a IS NULL AND u IS NULL) AS ok
     FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
       CREATE (:Agent {agentId: \$agentId, name: \$name, displayName: \$name}))
     RETURN ok AS created, a IS NOT NULL AS existed, u IS NOT NULL AS collided" \
    --compact >/dev/null
done

echo ""
echo "Agent roster seeded into ${G} (idempotent)."
echo "Verify with:"
echo "  redis-cli -h ${HOST} -p ${PORT} GRAPH.QUERY ${G} \"MATCH (a:Agent) RETURN a.agentId ORDER BY a.agentId\""
