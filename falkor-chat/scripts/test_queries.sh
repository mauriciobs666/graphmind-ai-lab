#!/usr/bin/env bash
# test_queries.sh — end-to-end test of every canonical query in docs/QUERIES.md.
#
# Usage:
#   ./scripts/test_queries.sh
#
# Uses an isolated graph (ws:test + reference) wiped at start and end.
# Exits non-zero on the first unexpected result.

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS="ws:test"
REF="reference"

PASS=0
FAIL=0

# ── helpers ──────────────────────────────────────────────────────────────────

gq()  { redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY    "$1" "$2"; }
rq()  { redis-cli -h "$HOST" -p "$PORT" GRAPH.RO_QUERY "$1" "$2"; }
gp()  { redis-cli -h "$HOST" -p "$PORT" GRAPH.PROFILE  "$1" "$2"; }

assert_contains() {
  local label="$1" expected="$2" actual="$3"
  if echo "$actual" | grep -qF "$expected"; then
    echo "  ✓ ${label}"
    PASS=$((PASS+1))
  else
    echo "  ✗ ${label}"
    echo "    expected to contain: ${expected}"
    echo "    got: ${actual}"
    FAIL=$((FAIL+1))
  fi
}

assert_not_contains() {
  local label="$1" unexpected="$2" actual="$3"
  if echo "$actual" | grep -qF "$unexpected"; then
    echo "  ✗ ${label}"
    echo "    expected NOT to contain: ${unexpected}"
    echo "    got: ${actual}"
    FAIL=$((FAIL+1))
  else
    echo "  ✓ ${label}"
    PASS=$((PASS+1))
  fi
}

assert_index_scan() {
  local label="$1" profile="$2"
  assert_contains "$label" "Node By Index Scan" "$profile"
  assert_not_contains "$label (no label scan)" "NodeByLabelScan" "$profile"
}

# ── setup ────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " falkor-chat query tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "▶ connectivity"
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG && echo "  ✓ FalkorDB reachable" || { echo "  ✗ cannot reach FalkorDB"; exit 1; }

echo ""
echo "▶ teardown (clean slate)"
redis-cli -h "$HOST" -p "$PORT" GRAPH.DELETE "$WS"  2>/dev/null || true
redis-cli -h "$HOST" -p "$PORT" GRAPH.DELETE "$REF" 2>/dev/null || true

echo ""
echo "▶ bootstrap schema"
EMBEDDING_DIM=4 ./scripts/bootstrap_schema.sh test 2>&1 | grep -E "done:|ERROR" || true

# ── §2: users & membership ───────────────────────────────────────────────────

echo ""
echo "▶ §2 users & membership"

out=$(gq "$WS" "MERGE (u:User {userId:'u1'}) ON CREATE SET u.displayName='Alice', u.email='alice@example.com', u.createdAt=1000 RETURN u.userId")
assert_contains "create user Alice" "u1" "$out"

out=$(gq "$WS" "MERGE (u:User {userId:'u2'}) ON CREATE SET u.displayName='Bob', u.email='bob@example.com', u.createdAt=1001 RETURN u.userId")
assert_contains "create user Bob" "u2" "$out"

# idempotency: merge same user again — no new node
out=$(gq "$WS" "MERGE (u:User {userId:'u1'}) ON CREATE SET u.displayName='Alice', u.email='alice@example.com', u.createdAt=1000 ON MATCH SET u.displayName='Alice' RETURN u.displayName")
assert_contains "merge user idempotent" "Alice" "$out"
assert_not_contains "merge user no duplicate node" "Nodes created: 2" "$out"

# uniqueness constraint: duplicate userId must fail
out=$(gq "$WS" "CREATE (:User {userId:'u1', displayName:'Imposter'})" 2>&1)
assert_contains "constraint blocks duplicate userId" "unique constraint violation" "$out"

# ── §3: channels & threads ───────────────────────────────────────────────────

echo ""
echo "▶ §3 channels & threads"

out=$(gq "$WS" "MERGE (c:Channel {channelId:'ch1'}) ON CREATE SET c.name='general', c.createdAt=1000 RETURN c.name")
assert_contains "create channel general" "general" "$out"

out=$(gq "$WS" "MERGE (c:Channel {channelId:'ch2'}) ON CREATE SET c.name='random', c.createdAt=1001 RETURN c.name")
assert_contains "create channel random" "random" "$out"

out=$(gq "$WS" "MATCH (u:User {userId:'u1'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (u)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='member', r.joinedAt=1000 RETURN r.role")
assert_contains "add Alice to general" "member" "$out"

out=$(gq "$WS" "MATCH (u:User {userId:'u2'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (u)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='member', r.joinedAt=1001 RETURN r.role")
assert_contains "add Bob to general" "member" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t:Thread) RETURN count(t) AS n")
assert_contains "no threads yet" "0" "$out"

# create two threads
out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) MERGE (t:Thread {threadId:'th1'}) ON CREATE SET t.title='First thread', t.createdAt=1000, t.updatedAt=1000 MERGE (c)-[:HAS_THREAD]->(t) RETURN t.threadId")
assert_contains "create thread th1" "th1" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) MERGE (t:Thread {threadId:'th2'}) ON CREATE SET t.title='Second thread', t.createdAt=1001, t.updatedAt=1001 MERGE (c)-[:HAS_THREAD]->(t) RETURN t.threadId")
assert_contains "create thread th2" "th2" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t:Thread) RETURN t.threadId ORDER BY t.updatedAt DESC LIMIT 10")
assert_contains "list threads returns th1" "th1" "$out"
assert_contains "list threads returns th2" "th2" "$out"

# ── §4: messages ─────────────────────────────────────────────────────────────

echo ""
echo "▶ §4 messages — write path"

# first message in th1
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'}) MATCH (author:User {userId:'u1'}) MERGE (m:Message {msgId:'m1'}) ON CREATE SET m.text='Hello thread', m.role='human', m.createdAt=2000 CREATE (t)-[:HEAD]->(m) CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2000 RETURN m.msgId")
assert_contains "post first message m1" "m1" "$out"

# second message
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) MATCH (author:User {userId:'u2'}) MERGE (m:Message {msgId:'m2'}) ON CREATE SET m.text='Hi Alice!', m.role='human', m.createdAt=2001 CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2001 RETURN m.msgId")
assert_contains "post second message m2" "m2" "$out"

# third message
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) MATCH (author:User {userId:'u1'}) MERGE (m:Message {msgId:'m3'}) ON CREATE SET m.text='Welcome Bob, great to have you here', m.role='human', m.createdAt=2002 CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2002 RETURN m.msgId")
assert_contains "post third message m3" "m3" "$out"

# reply edge
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) MATCH (author:User {userId:'u2'}) MATCH (quoted:Message {msgId:'m3'}) MERGE (m:Message {msgId:'m4'}) ON CREATE SET m.text='Thanks Alice!', m.role='human', m.createdAt=2003 CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) CREATE (m)-[:REPLY_TO]->(quoted) SET t.updatedAt=2003 RETURN m.msgId")
assert_contains "post reply m4 with REPLY_TO" "m4" "$out"

# verify TAIL points to m4
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[:TAIL]->(tail:Message) RETURN tail.msgId")
assert_contains "TAIL points to last message" "m4" "$out"

# verify HEAD still points to m1
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[:HEAD]->(head:Message) RETURN head.msgId")
assert_contains "HEAD still points to first message" "m1" "$out"

# uniqueness: duplicate msgId must fail
out=$(gq "$WS" "CREATE (:Message {msgId:'m1', text:'duplicate'})" 2>&1)
assert_contains "constraint blocks duplicate msgId" "unique constraint violation" "$out"

echo ""
echo "▶ §4 messages — read path"

# read full thread in order
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId, m.text, m.createdAt, author.displayName ORDER BY m.createdAt")
assert_contains "thread read: m1 present" "m1" "$out"
assert_contains "thread read: m2 present" "m2" "$out"
assert_contains "thread read: m3 present" "m3" "$out"
assert_contains "thread read: m4 present" "m4" "$out"
assert_contains "thread read: Alice present" "Alice" "$out"
assert_contains "thread read: Bob present" "Bob" "$out"

# verify order: m1 appears before m4 in output
m1_line=$(echo "$out" | grep -n "m1" | head -1 | cut -d: -f1)
m4_line=$(echo "$out" | grep -n "m4" | head -1 | cut -d: -f1)
if [ "$m1_line" -lt "$m4_line" ]; then
  echo "  ✓ thread read: messages in chronological order"
  PASS=$((PASS+1))
else
  echo "  ✗ thread read: wrong order (m1 line ${m1_line}, m4 line ${m4_line})"
  FAIL=$((FAIL+1))
fi

# cursor pagination: messages after m2
out=$(gq "$WS" "MATCH (cursor:Message {msgId:'m2'})-[:NEXT*1..]->(m:Message) RETURN m.msgId ORDER BY m.createdAt LIMIT 10")
assert_contains     "cursor pagination: m3 after m2" "m3" "$out"
assert_contains     "cursor pagination: m4 after m2" "m4" "$out"
assert_not_contains "cursor pagination: m1 not in result" "m1" "$out"
assert_not_contains "cursor pagination: m2 not in result" "m2" "$out"

# get single message
out=$(gq "$WS" "MATCH (m:Message {msgId:'m4'}) OPTIONAL MATCH (m)-[:POSTED_BY]->(author) OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message) RETURN m.text, author.displayName, quoted.msgId AS quotedId")
assert_contains "get single message m4: text" "Thanks Alice" "$out"
assert_contains "get single message m4: author" "Bob" "$out"
assert_contains "get single message m4: quotedId" "m3" "$out"

# thread.updatedAt updated after messages
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'}) RETURN t.updatedAt")
assert_contains "thread updatedAt bumped to last msg createdAt" "2003" "$out"

echo ""
echo "▶ §3 list recent threads (after messages posted)"
out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t:Thread) RETURN t.threadId, t.updatedAt ORDER BY t.updatedAt DESC LIMIT 10")
assert_contains "recent threads: th1 most recent" "th1" "$out"

# ── §5: full-text search ─────────────────────────────────────────────────────

echo ""
echo "▶ §5 full-text search"

# allow a moment for the fulltext index to pick up the nodes
sleep 1

out=$(gq "$WS" "CALL db.idx.fulltext.queryNodes('Message', 'welcome') YIELD node AS m, score RETURN m.msgId, m.text, score ORDER BY score DESC LIMIT 5" 2>&1)
assert_contains "fulltext: 'welcome' finds m3" "m3" "$out"
assert_not_contains "fulltext: 'welcome' no error" "ERR" "$out"

out=$(gq "$WS" "CALL db.idx.fulltext.queryNodes('Message', 'Hello') YIELD node AS m, score RETURN m.msgId ORDER BY score DESC LIMIT 5" 2>&1)
assert_contains "fulltext: 'Hello' finds m1" "m1" "$out"

# ── §6: GraphRAG vector retrieval ────────────────────────────────────────────

echo ""
echo "▶ §6 GraphRAG — set embeddings and query"

# set dummy 4-dim embeddings on messages
gq "$WS" "MATCH (m:Message {msgId:'m1'}) SET m.embedding = vecf32([1.0, 0.0, 0.0, 0.0])" > /dev/null
gq "$WS" "MATCH (m:Message {msgId:'m2'}) SET m.embedding = vecf32([0.9, 0.1, 0.0, 0.0])" > /dev/null
gq "$WS" "MATCH (m:Message {msgId:'m3'}) SET m.embedding = vecf32([0.0, 0.0, 1.0, 0.0])" > /dev/null
gq "$WS" "MATCH (m:Message {msgId:'m4'}) SET m.embedding = vecf32([0.0, 0.0, 0.9, 0.1])" > /dev/null

# query nearest to [1.0, 0.0, 0.0, 0.0] → should rank m1 first (score 0), m2 close
out=$(gq "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score RETURN seed.msgId, score ORDER BY score ASC")
assert_contains "vector query: m1 in top-2" "m1" "$out"
assert_contains "vector query: m2 in top-2" "m2" "$out"
assert_not_contains "vector query: m3 not in top-2 (different direction)" "m3" "$out"

# score for identical vector should be 0
m1_score=$(echo "$out" | grep -A1 "m1" | tail -1 | tr -d ' ')
if [ "$m1_score" = "0" ]; then
  echo "  ✓ vector query: identical vector scores 0 (cosine distance)"
  PASS=$((PASS+1))
else
  echo "  ✗ vector query: expected score 0 for identical vector, got '${m1_score}'"
  FAIL=$((FAIL+1))
fi

# hybrid retrieval: vector seed + channel scope traversal
out=$(gq "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId, seed.text, score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT 5" 2>&1)
assert_contains "hybrid retrieval: returns results" "m1" "$out"
assert_not_contains "hybrid retrieval: no error" "ERR" "$out"

# ── §7: agents ───────────────────────────────────────────────────────────────

echo ""
echo "▶ §7 agents"

out=$(gq "$WS" "MERGE (a:Agent {agentId:'bot1'}) ON CREATE SET a.name='FalkorBot', a.model='gpt-4o', a.createdAt=1000 RETURN a.agentId")
assert_contains "create agent bot1" "bot1" "$out"

out=$(gq "$WS" "MATCH (a:Agent {agentId:'bot1'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (a)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='assistant', r.joinedAt=1000 RETURN r.role")
assert_contains "add agent to channel" "assistant" "$out"

out=$(gq "$WS" "MATCH (u)-[:MEMBER_OF]->(c:Channel {channelId:'ch1'}) RETURN coalesce(u.userId, u.agentId) AS memberId, u.displayName AS displayName, labels(u) AS type ORDER BY u.displayName")
assert_contains "channel members: Alice" "u1" "$out"
assert_contains "channel members: Bob" "u2" "$out"
assert_contains "channel members: bot1" "bot1" "$out"
assert_contains "channel members: Agent label visible" "Agent" "$out"

# agent posts a message (assistant role)
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) MATCH (author:Agent {agentId:'bot1'}) MERGE (m:Message {msgId:'m5'}) ON CREATE SET m.text='I am the AI assistant', m.role='assistant', m.createdAt=3000 CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=3000 RETURN m.role")
assert_contains "agent posts assistant message" "assistant" "$out"

# ── §8: diagnostics / index usage ────────────────────────────────────────────

echo ""
echo "▶ §8 index usage (GRAPH.PROFILE)"

prof=$(gp "$WS" "MATCH (m:Message {msgId:'m1'}) RETURN m")
assert_index_scan "Message lookup by msgId uses index" "$prof"

prof=$(gp "$WS" "MATCH (u:User {userId:'u1'}) RETURN u")
assert_index_scan "User lookup by userId uses index" "$prof"

prof=$(gp "$WS" "MATCH (t:Thread {threadId:'th1'}) RETURN t")
assert_index_scan "Thread lookup by threadId uses index" "$prof"

prof=$(gp "$WS" "MATCH (c:Channel {channelId:'ch1'}) RETURN c")
assert_index_scan "Channel lookup by channelId uses index" "$prof"

# ── §ref: reference graph ────────────────────────────────────────────────────

echo ""
echo "▶ reference graph — WorkflowDef composite constraint"

out=$(gq "$REF" "CREATE (:WorkflowDef {key:'onboard', version:'1', name:'Onboarding', kind:'process'})" 2>&1)
assert_contains "create WorkflowDef v1" "Nodes created: 1" "$out"

# same key+version must fail
out=$(gq "$REF" "CREATE (:WorkflowDef {key:'onboard', version:'1', name:'Dupe'})" 2>&1)
assert_contains "composite constraint blocks duplicate key+version" "unique constraint violation" "$out"

# different version is allowed
out=$(gq "$REF" "CREATE (:WorkflowDef {key:'onboard', version:'2', name:'Onboarding v2', kind:'process'})" 2>&1)
assert_contains "different version allowed" "Nodes created: 1" "$out"

# ── teardown ─────────────────────────────────────────────────────────────────

echo ""
echo "▶ teardown"
redis-cli -h "$HOST" -p "$PORT" GRAPH.DELETE "$WS"  > /dev/null && echo "  ✓ ws:test deleted"
redis-cli -h "$HOST" -p "$PORT" GRAPH.DELETE "$REF" > /dev/null && echo "  ✓ reference deleted"

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TOTAL=$((PASS+FAIL))
echo " Results: ${PASS}/${TOTAL} passed"
if [ "$FAIL" -gt 0 ]; then
  echo " FAILED: ${FAIL} assertion(s)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
