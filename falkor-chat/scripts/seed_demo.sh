#!/usr/bin/env bash
# seed_demo.sh — register the AI agent + a minimal demo (channel + thread) so a
# human can open the web UI, post into a real channel, and @mention the agent.
#
# Usage:
#   ./scripts/seed_demo.sh [<workspaceId>]        # default: $FALKORCHAT_WS_ID or "acme"
#
# What it seeds (all idempotent — safe to re-run):
#   * User  {userId}          — the hardcoded M1 tenant actor (config.USER_ID)
#   * Agent {agentId}         — the responder posts as this Agent; @mentioning it
#                               (by agentId) triggers a reply. Wire the SAME id into
#                               AgentResponder via FALKORCHAT_AGENT_ID.
#   * Channel {channelId}     — a demo channel (fixed id → MERGE, backed by the
#   * Thread  {threadId}        Channel/Thread uniqueness constraints)
#   * MEMBER_OF edges         — user + agent as channel members (roster)
#
# NOTE — unlike the app's runtime channel/thread creates (server-minted uuids,
# plain CREATE, non-idempotent), this seed uses FIXED ids + MERGE on purpose so a
# re-run never duplicates the demo. Both MERGEs are backed by the uniqueness
# constraints created by bootstrap_schema.sh — run that FIRST.
#
# Mention resolution only needs the Agent NODE to exist (services.resolve_member_kinds
# looks up agentId, not channel membership); MEMBER_OF is seeded for roster/scoping.
#
# Env vars (all optional):
#   FALKORDB_HOST        (default: 127.0.0.1)
#   FALKORDB_PORT        (default: 6379)
#   FALKORCHAT_WS_ID     (default: acme)     — workspace id (graph key ws:<id>)
#   FALKORCHAT_USER_ID   (default: u1)       — must match config.USER_ID
#   FALKORCHAT_USER_NAME (default: Demo User)
#   FALKORCHAT_AGENT_ID  (default: assistant)— must match config.AGENT_ID
#   FALKORCHAT_AGENT_NAME(default: Assistant)
#   DEMO_CHANNEL_ID      (default: demo-general)
#   DEMO_CHANNEL_NAME    (default: general)
#   DEMO_THREAD_ID       (default: demo-welcome)
#   DEMO_THREAD_TITLE    (default: Welcome)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
USER_ID="${FALKORCHAT_USER_ID:-u1}"
USER_NAME="${FALKORCHAT_USER_NAME:-Demo User}"
AGENT_ID="${FALKORCHAT_AGENT_ID:-assistant}"
AGENT_NAME="${FALKORCHAT_AGENT_NAME:-Assistant}"
CHANNEL_ID="${DEMO_CHANNEL_ID:-demo-general}"
CHANNEL_NAME="${DEMO_CHANNEL_NAME:-general}"
THREAD_ID="${DEMO_THREAD_ID:-demo-welcome}"
THREAD_TITLE="${DEMO_THREAD_TITLE:-Welcome}"

G="ws:${WS_ID}"

gq() {
  # Parameterised GRAPH.QUERY — never interpolate values into the Cypher body.
  redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" "$1" --compact >/dev/null
}

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}

echo "── seeding demo into ${G} ─────────────────────────────────"

# User (guarded ensure, QUERIES.md §2 — namespace-unique across User/Agent).
echo "[1/5] User  ${USER_ID} (${USER_NAME})"
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
  "CYPHER userId='${USER_ID}' displayName='${USER_NAME}'
   OPTIONAL MATCH (u:User {userId: \$userId})
   OPTIONAL MATCH (a:Agent {agentId: \$userId})
   WITH u, a, (u IS NULL AND a IS NULL) AS ok
   FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
     CREATE (:User {userId: \$userId, displayName: \$displayName}))
   RETURN ok AS created" --compact >/dev/null

# Agent (guarded ensure, QUERIES.md §7).
echo "[2/5] Agent ${AGENT_ID} (${AGENT_NAME})"
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
  "CYPHER agentId='${AGENT_ID}' name='${AGENT_NAME}'
   OPTIONAL MATCH (a:Agent {agentId: \$agentId})
   OPTIONAL MATCH (u:User {userId: \$agentId})
   WITH a, u, (a IS NULL AND u IS NULL) AS ok
   FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
     CREATE (:Agent {agentId: \$agentId, name: \$name, displayName: \$name}))
   RETURN ok AS created" --compact >/dev/null

# Channel (fixed id → MERGE, backed by the Channel.channelId uniqueness constraint).
echo "[3/5] Channel ${CHANNEL_ID} (#${CHANNEL_NAME})"
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
  "CYPHER channelId='${CHANNEL_ID}' name='${CHANNEL_NAME}'
   MERGE (c:Channel {channelId: \$channelId})
   ON CREATE SET c.name = \$name, c.createdAt = timestamp()
   RETURN c.channelId" --compact >/dev/null

# Thread under the channel (fixed id → MERGE, backed by Thread.threadId constraint).
echo "[4/5] Thread ${THREAD_ID} (\"${THREAD_TITLE}\")"
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
  "CYPHER channelId='${CHANNEL_ID}' threadId='${THREAD_ID}' title='${THREAD_TITLE}'
   MATCH (c:Channel {channelId: \$channelId})
   MERGE (t:Thread {threadId: \$threadId})
   ON CREATE SET t.title = \$title, t.createdAt = timestamp(), t.updatedAt = timestamp()
   MERGE (c)-[:HAS_THREAD]->(t)
   RETURN t.threadId" --compact >/dev/null

# Membership: user + agent as channel members (roster / future scoping).
echo "[5/5] MEMBER_OF: ${USER_ID}, ${AGENT_ID} -> ${CHANNEL_ID}"
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$G" \
  "CYPHER userId='${USER_ID}' agentId='${AGENT_ID}' channelId='${CHANNEL_ID}'
   MATCH (c:Channel {channelId: \$channelId})
   MATCH (u:User {userId: \$userId})
   MATCH (a:Agent {agentId: \$agentId})
   MERGE (u)-[:MEMBER_OF]->(c)
   MERGE (a)-[:MEMBER_OF]->(c)
   RETURN c.channelId" --compact >/dev/null

echo ""
echo "Demo seeded (idempotent). Open the web UI, pick #${CHANNEL_NAME} -> \"${THREAD_TITLE}\","
echo "and post e.g.:  @${AGENT_ID} what is falkor-chat?"
