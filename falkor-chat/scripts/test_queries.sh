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

# v2 guarded ensures (DEF-1): canonical bodies in QUERIES.md §2/§7; as with §4, the
# test copies keep every clause verbatim except the final RETURN, which composes the
# three status booleans into one grep-able string ("c=… e=… col=…").

out=$(gq "$WS" "OPTIONAL MATCH (u:User {userId:'u1'}) OPTIONAL MATCH (a:Agent {agentId:'u1'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'u1', displayName:'Alice', email:'alice@example.com'})) RETURN 'c='+toString(ok)+' e='+toString(u IS NOT NULL)+' col='+toString(a IS NOT NULL) AS status")
assert_contains "ensure user Alice (v2 fresh create)" "c=true e=false col=false" "$out"

out=$(gq "$WS" "OPTIONAL MATCH (u:User {userId:'u2'}) OPTIONAL MATCH (a:Agent {agentId:'u2'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'u2', displayName:'Bob', email:'bob@example.com'})) RETURN 'c='+toString(ok)+' e='+toString(u IS NOT NULL)+' col='+toString(a IS NOT NULL) AS status")
assert_contains "ensure user Bob (v2 fresh create)" "c=true e=false col=false" "$out"

# idempotency: re-ensure same user — status row says existed, nothing written
out=$(gq "$WS" "OPTIONAL MATCH (u:User {userId:'u1'}) OPTIONAL MATCH (a:Agent {agentId:'u1'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'u1', displayName:'Alice', email:'alice@example.com'})) RETURN 'c='+toString(ok)+' e='+toString(u IS NOT NULL)+' col='+toString(a IS NOT NULL) AS status")
assert_contains "re-ensure user idempotent (existed, no write)" "c=false e=true col=false" "$out"
assert_not_contains "re-ensure user creates no node" "Nodes created" "$out"

out=$(gq "$WS" "MATCH (u:User {userId:'u1'}) RETURN count(u) AS n")
assert_contains "re-ensure left exactly one Alice node" "1" "$out"

# uniqueness constraint: duplicate userId must fail
out=$(gq "$WS" "CREATE (:User {userId:'u1', displayName:'Imposter'})" 2>&1)
assert_contains "constraint blocks duplicate userId" "unique constraint violation" "$out"

# ── §3: channels & threads ───────────────────────────────────────────────────

echo ""
echo "▶ §3 channels & threads"

# plain CREATE (K-007): ids are server-minted, MERGE could never match; the
# uniqueness constraint is the backstop — creates are non-idempotent.
out=$(gq "$WS" "CREATE (c:Channel {channelId:'ch1', name:'general', createdAt:1000}) RETURN c.name")
assert_contains "create channel general" "general" "$out"

out=$(gq "$WS" "CREATE (c:Channel {channelId:'ch2', name:'random', createdAt:1001}) RETURN c.name")
assert_contains "create channel random" "random" "$out"

out=$(gq "$WS" "CREATE (:Channel {channelId:'ch1', name:'imposter'})" 2>&1)
assert_contains "constraint blocks duplicate channelId" "unique constraint violation" "$out"

# list channels in the workspace (newest-first by createdAt, index-anchored on channelId)
out=$(gq "$WS" "MATCH (c:Channel) WHERE c.channelId > '' RETURN c.channelId AS channelId, c.name AS name, c.createdAt AS createdAt ORDER BY c.createdAt DESC LIMIT 100")
# ch2 (createdAt 1001) newest-first before ch1 (createdAt 1000); both present
ch2_line=$(echo "$out" | grep -n "ch2" | head -1 | cut -d: -f1)
ch1_line=$(echo "$out" | grep -n "ch1" | head -1 | cut -d: -f1)
if echo "$out" | grep -qF "general" && echo "$out" | grep -qF "random" \
   && [ -n "$ch2_line" ] && [ -n "$ch1_line" ] && [ "$ch2_line" -lt "$ch1_line" ]; then
  echo "  ✓ list channels: both channels listed, newest-first (ch2 before ch1)"
  PASS=$((PASS+1))
else
  echo "  ✗ list channels: missing channel or wrong order"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi

out=$(gq "$WS" "MATCH (u:User {userId:'u1'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (u)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='member', r.joinedAt=1000 RETURN r.role")
assert_contains "add Alice to general" "member" "$out"

out=$(gq "$WS" "MATCH (u:User {userId:'u2'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (u)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='member', r.joinedAt=1001 RETURN r.role")
assert_contains "add Bob to general" "member" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t:Thread) RETURN count(t) AS n")
assert_contains "no threads yet" "0" "$out"

# create two threads (plain CREATE anchored on the channel; zero rows = nothing written)
out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) CREATE (t:Thread {threadId:'th1', title:'First thread', createdAt:1000, updatedAt:1000}) CREATE (c)-[:HAS_THREAD]->(t) RETURN t.threadId")
assert_contains "create thread th1" "th1" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) CREATE (t:Thread {threadId:'th2', title:'Second thread', createdAt:1001, updatedAt:1001}) CREATE (c)-[:HAS_THREAD]->(t) RETURN t.threadId")
assert_contains "create thread th2" "th2" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) CREATE (:Thread {threadId:'th1', title:'dupe'})" 2>&1)
assert_contains "constraint blocks duplicate threadId" "unique constraint violation" "$out"

out=$(gq "$WS" "MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t:Thread) RETURN t.threadId ORDER BY t.updatedAt DESC LIMIT 10")
assert_contains "list threads returns th1" "th1" "$out"
assert_contains "list threads returns th2" "th2" "$out"

# ── §4: messages (v2 self-guarding write paths) ──────────────────────────────
#
# The canonical v2 bodies live in docs/QUERIES.md §4. The test copies keep every
# clause verbatim except the final RETURN, which composes the four status
# booleans into one grep-able string ("w=… hh=… dup=… auth=…") — same values,
# assert-friendly shape.

echo ""
echo "▶ §4 messages — write path (v2)"

# first message in th1 (v2 first-path; status row + guarded CREATE)
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'}) OPTIONAL MATCH (t)-[:HEAD]->(h) OPTIONAL MATCH (dup:Message {msgId:'m1'}) OPTIONAL MATCH (ua:User {userId:'u1'}) OPTIONAL MATCH (aa:Agent {agentId:'u1'}) WITH t, h, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, h, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, h, dup, author, mems, (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m1', text:'Hello thread', role:'user', createdAt:2000, threadId:'th1'}) CREATE (t)-[:HEAD]->(m) CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2000 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh='+toString(h IS NOT NULL)+' dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "post first message m1: status row written" "w=true hh=false dup=false auth=true" "$out"

out=$(gq "$WS" "MATCH (m:Message {msgId:'m1'}) RETURN m.msgId")
assert_contains "post first message m1: node exists" "m1" "$out"

# second message (v2 subsequent-path)
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'m2'}) OPTIONAL MATCH (ua:User {userId:'u2'}) OPTIONAL MATCH (aa:Agent {agentId:'u2'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m2', text:'Hi Alice!', role:'user', createdAt:2001, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2001 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh=false dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "post second message m2 (v2 subsequent)" "w=true hh=false dup=false auth=true" "$out"

# third message
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'m3'}) OPTIONAL MATCH (ua:User {userId:'u1'}) OPTIONAL MATCH (aa:Agent {agentId:'u1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m3', text:'Welcome Bob, great to have you here', role:'user', createdAt:2002, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2002 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh=false dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "post third message m3 (v2 subsequent)" "w=true hh=false dup=false auth=true" "$out"

# reply m4 — REPLY_TO created INSIDE the guarded FOREACH (OQ4 live verification)
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) MATCH (quoted:Message {msgId:'m3'}) OPTIONAL MATCH (dup:Message {msgId:'m4'}) OPTIONAL MATCH (ua:User {userId:'u2'}) OPTIONAL MATCH (aa:Agent {agentId:'u2'}) WITH t, tailRel, prev, quoted, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, quoted, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, quoted, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m4', text:'Thanks Alice!', role:'user', createdAt:2003, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) CREATE (m)-[:REPLY_TO]->(quoted) SET t.updatedAt=2003 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh=false dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "post reply m4 (v2 subsequent + REPLY_TO in guard)" "w=true hh=false dup=false auth=true" "$out"

out=$(gq "$WS" "MATCH (:Message {msgId:'m4'})-[r:REPLY_TO]->(q:Message) RETURN count(r) AS n, q.msgId")
assert_contains "m4 REPLY_TO edge exists (created inside guarded FOREACH)" "m3" "$out"

# verify TAIL points to m4
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[:TAIL]->(tail:Message) RETURN tail.msgId")
assert_contains "TAIL points to last message" "m4" "$out"

# verify HEAD still points to m1
out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[:HEAD]->(head:Message) RETURN head.msgId")
assert_contains "HEAD still points to first message" "m1" "$out"

# uniqueness: duplicate msgId must fail (constraint = concurrency backstop)
out=$(gq "$WS" "CREATE (:Message {msgId:'m1', text:'duplicate'})" 2>&1)
assert_contains "constraint blocks duplicate msgId" "unique constraint violation" "$out"

echo ""
echo "▶ §4 write-path v2 guards (retry replay / first-post race / unknown author)"

# defect A regression: exact replay of the m2 subsequent write is a structural no-op
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'m2'}) OPTIONAL MATCH (ua:User {userId:'u2'}) OPTIONAL MATCH (aa:Agent {agentId:'u2'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m2', text:'Hi Alice!', role:'user', createdAt:2001, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2001 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh=false dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "replay of subsequent write reports dupMsg=true (idempotent)" "w=false hh=false dup=true auth=true" "$out"

out=$(gq "$WS" "MATCH (a:Message)-[r:NEXT]->(b:Message) RETURN 'next='+toString(count(r))+' self='+toString(sum(CASE WHEN a.msgId = b.msgId THEN 1 ELSE 0 END)) AS s")
assert_contains "replay left NEXT chain intact (3 edges, no self-loop)" "next=3 self=0" "$out"

out=$(gq "$WS" "MATCH (m:Message) RETURN count(m) AS n")
assert_contains "replay created no message node (still 4)" "4" "$out"

# defect B regression: late first-post on a headed thread refuses (hadHead=true)
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'}) OPTIONAL MATCH (t)-[:HEAD]->(h) OPTIONAL MATCH (dup:Message {msgId:'mX'}) OPTIONAL MATCH (ua:User {userId:'u1'}) OPTIONAL MATCH (aa:Agent {agentId:'u1'}) WITH t, h, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, h, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, h, dup, author, mems, (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'mX', text:'late first', role:'user', createdAt:2004, threadId:'th1'}) CREATE (t)-[:HEAD]->(m) CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2004 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh='+toString(h IS NOT NULL)+' dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "late first-post refused with hadHead=true" "w=false hh=true dup=false auth=true" "$out"

out=$(gq "$WS" "MATCH (t:Thread {threadId:'th1'})-[r:HEAD]->() RETURN count(r) AS n")
assert_contains "thread still has exactly one HEAD" "1" "$out"

# unknown author refuses with authorFound=false, nothing written
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'mY'}) OPTIONAL MATCH (ua:User {userId:'ghost'}) OPTIONAL MATCH (aa:Agent {agentId:'ghost'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'mY', text:'ghost post', role:'user', createdAt:2005, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2005 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' hh=false dup='+toString(dup IS NOT NULL)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "unknown author refused with authorFound=false" "w=false hh=false dup=false auth=false" "$out"

out=$(gq "$WS" "OPTIONAL MATCH (m:Message {msgId:'mY'}) RETURN 'exists='+toString(m IS NOT NULL) AS s")
assert_contains "unknown-author message was not written" "exists=false" "$out"

# threadId denorm: v2 write stamped it inline
out=$(gq "$WS" "MATCH (m:Message {msgId:'m1'}) RETURN m.threadId")
assert_contains "v2 write stamped threadId on m1" "th1" "$out"

# v2 subsequent write plans with indexed anchors only (dup msgId → guarded no-op)
prof=$(gp "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'m2'}) OPTIONAL MATCH (ua:User {userId:'u2'}) OPTIONAL MATCH (aa:Agent {agentId:'u2'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m2', text:'Hi Alice!', role:'user', createdAt:2001, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=2001 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN ok")
assert_not_contains "v2 subsequent write has no All Node Scan" "All Node Scan" "$prof"

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
echo "▶ §4.x threadId backfill (one-off ops query)"

# simulate pre-K-007 rows: strip threadId from m2 and m3
gq "$WS" "MATCH (m:Message) WHERE m.msgId IN ['m2','m3'] SET m.threadId = null" > /dev/null

out=$(gq "$WS" "MATCH (t:Thread)-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.threadId IS NULL SET m.threadId = t.threadId RETURN 'backfilled='+toString(count(m)) AS s")
assert_contains "backfill run 1 stamps the 2 stripped messages" "backfilled=2" "$out"

out=$(gq "$WS" "MATCH (t:Thread)-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.threadId IS NULL SET m.threadId = t.threadId RETURN 'backfilled='+toString(count(m)) AS s")
assert_contains "backfill run 2 is idempotent (0)" "backfilled=0" "$out"

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

# §5 RETURN carries threadId (denormalized navigation metadata)
out=$(gq "$WS" "CALL db.idx.fulltext.queryNodes('Message', 'welcome') YIELD node AS m, score RETURN m.msgId, m.threadId AS threadId, m.text, m.createdAt, score ORDER BY score DESC LIMIT 5" 2>&1)
assert_contains "fulltext rows carry threadId" "th1" "$out"

# ── §6: GraphRAG vector retrieval ────────────────────────────────────────────
#
# Exercised here at the isolated-suite dim (EMBEDDING_DIM=4). The dim-1024
# verification (real embedding size, wrong-dim quirk, RAM line) lives in the
# graph-dba deliverable docs/archive/plans/m2-graphrag.md — do not re-encode 1024 here.
# Score is cosine distance: 0 = identical, ASC = most similar first (DESIGN §1.3).
# There is NO Entity pipeline in M2 — the §6 Entity expansion must no-op cleanly.

echo ""
echo "▶ §6 GraphRAG — set embeddings and query"

# §6 set-embedding write path (SET m.embedding = vecf32($embedding)) — must commit
out=$(gq "$WS" "MATCH (m:Message {msgId:'m1'}) SET m.embedding = vecf32([1.0, 0.0, 0.0, 0.0]) RETURN m.msgId")
assert_contains "§6 set-embedding write commits (Properties set)" "Properties set" "$out"
gq "$WS" "MATCH (m:Message {msgId:'m2'}) SET m.embedding = vecf32([0.9, 0.1, 0.0, 0.0])" > /dev/null
gq "$WS" "MATCH (m:Message {msgId:'m3'}) SET m.embedding = vecf32([0.0, 0.0, 1.0, 0.0])" > /dev/null
gq "$WS" "MATCH (m:Message {msgId:'m4'}) SET m.embedding = vecf32([0.0, 0.0, 0.9, 0.1])" > /dev/null

# ANN retrieval (route via RO_QUERY): nearest to [1,0,0,0] → m1 first (score 0), m2 close
out=$(rq "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score RETURN seed.msgId, score ORDER BY score ASC")
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

# ranking is strictly ASC by distance: m1 (identical) sorts before m2 (near)
m1_line=$(echo "$out" | grep -n "m1" | head -1 | cut -d: -f1)
m2_line=$(echo "$out" | grep -n "m2" | head -1 | cut -d: -f1)
if [ -n "$m1_line" ] && [ -n "$m2_line" ] && [ "$m1_line" -lt "$m2_line" ]; then
  echo "  ✓ §6 ANN ranks by cosine distance ASC (m1 identical before m2 near)"
  PASS=$((PASS+1))
else
  echo "  ✗ §6 ANN ranking wrong (m1 line ${m1_line}, m2 line ${m2_line})"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi

# full §6 hybrid retrieval: vector seed + thread/channel scope + Entity expansion
out=$(rq "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId, seed.text, seed.role, score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT 5" 2>&1)
assert_contains "hybrid retrieval: returns results" "m1" "$out"
assert_contains "§6 hybrid returns seed text" "Hello thread" "$out"
assert_not_contains "hybrid retrieval: no error" "ERR" "$out"

# Entity expansion no-ops cleanly: there is no Entity pipeline in M2, so
# relatedContext must be the empty list [] (not an error, not missing).
assert_contains "§6 Entity expansion no-ops (empty relatedContext [])" "[]" "$out"
out=$(rq "$WS" "MATCH (e:Entity) RETURN count(e) AS entities")
assert_contains "§6 Entity graph is empty (no entity pipeline in M2)" "0" "$out"

# PROFILE: the ANN retrieval must anchor on the vector index (ProcedureCall),
# never a whole-graph scan of Message.
prof=$(gp "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) MATCH (c:Channel {channelId:'ch1'})-[:HAS_THREAD]->(t) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId, score ORDER BY score ASC LIMIT 5")
assert_contains     "§6 ANN retrieval uses the vector index (ProcedureCall)" "ProcedureCall" "$prof"
assert_not_contains "§6 ANN retrieval has no All Node Scan"                  "All Node Scan" "$prof"

# workspace-wide variant (channel MATCH omitted) still retrieves via the index
out=$(rq "$WS" "CALL db.idx.vector.queryNodes('Message', 'embedding', 2, vecf32([1.0, 0.0, 0.0, 0.0])) YIELD node AS seed, score OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId, score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT 5" 2>&1)
assert_contains     "§6 workspace-wide variant returns m1" "m1" "$out"
assert_not_contains "§6 workspace-wide variant no error"   "ERR" "$out"

# ── §7: agents ───────────────────────────────────────────────────────────────

echo ""
echo "▶ §7 agents"

out=$(gq "$WS" "OPTIONAL MATCH (a:Agent {agentId:'bot1'}) OPTIONAL MATCH (u:User {userId:'bot1'}) WITH a, u, (a IS NULL AND u IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:Agent {agentId:'bot1', name:'FalkorBot', model:'gpt-4o', createdAt:1000})) RETURN 'c='+toString(ok)+' e='+toString(a IS NOT NULL)+' col='+toString(u IS NOT NULL) AS status")
assert_contains "ensure agent bot1 (v2 fresh create)" "c=true e=false col=false" "$out"

out=$(gq "$WS" "MATCH (a:Agent {agentId:'bot1'}) RETURN a.agentId")
assert_contains "agent bot1 node exists" "bot1" "$out"

# ── §2/§7 guarded-ensure cross-label guards (DEF-1) ──────────────────────────

echo ""
echo "▶ §2/§7 member-id namespace guards (DEF-1)"

# idempotent re-ensure of the agent — existed, nothing written
out=$(gq "$WS" "OPTIONAL MATCH (a:Agent {agentId:'bot1'}) OPTIONAL MATCH (u:User {userId:'bot1'}) WITH a, u, (a IS NULL AND u IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:Agent {agentId:'bot1', name:'FalkorBot', model:'gpt-4o', createdAt:1000})) RETURN 'c='+toString(ok)+' e='+toString(a IS NOT NULL)+' col='+toString(u IS NOT NULL) AS status")
assert_contains "re-ensure agent idempotent (existed, no write)" "c=false e=true col=false" "$out"
assert_not_contains "re-ensure agent creates no node" "Nodes created" "$out"

# DEF-1 repro direction 1: ensure_user on an id held by an Agent → refused, nothing written
out=$(gq "$WS" "OPTIONAL MATCH (u:User {userId:'bot1'}) OPTIONAL MATCH (a:Agent {agentId:'bot1'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'bot1', displayName:'Shadow', email:'shadow@example.com'})) RETURN 'c='+toString(ok)+' e='+toString(u IS NOT NULL)+' col='+toString(a IS NOT NULL) AS status")
assert_contains "ensure_user blocked by existing Agent (collided)" "c=false e=false col=true" "$out"

out=$(gq "$WS" "MATCH (u:User {userId:'bot1'}) RETURN count(u) AS n")
assert_contains "no shadow User written for bot1" "0" "$out"

# DEF-1 repro direction 2: ensure_agent on an id held by a User → refused, nothing written
out=$(gq "$WS" "OPTIONAL MATCH (a:Agent {agentId:'u1'}) OPTIONAL MATCH (u:User {userId:'u1'}) WITH a, u, (a IS NULL AND u IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:Agent {agentId:'u1', name:'EvilTwin', model:'x', createdAt:1}) ) RETURN 'c='+toString(ok)+' e='+toString(a IS NOT NULL)+' col='+toString(u IS NOT NULL) AS status")
assert_contains "ensure_agent blocked by existing User (collided)" "c=false e=false col=true" "$out"

out=$(gq "$WS" "MATCH (a:Agent {agentId:'u1'}) RETURN count(a) AS n")
assert_contains "no shadow Agent written for u1" "0" "$out"

# status-row shape on pre-guard corrupted state: both labels hold the id → existed AND collided
gq "$WS" "CREATE (:User {userId:'shdw'}), (:Agent {agentId:'shdw'})" > /dev/null
out=$(gq "$WS" "OPTIONAL MATCH (u:User {userId:'shdw'}) OPTIONAL MATCH (a:Agent {agentId:'shdw'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'shdw', displayName:'x', email:'x'})) RETURN 'c='+toString(ok)+' e='+toString(u IS NOT NULL)+' col='+toString(a IS NOT NULL) AS status")
assert_contains "corrupted both-exist state flagged (existed AND collided)" "c=false e=true col=true" "$out"
gq "$WS" "MATCH (n) WHERE n.userId = 'shdw' OR n.agentId = 'shdw' DELETE n" > /dev/null

# PROFILE: both cross-label existence checks must be index scans
prof=$(gp "$WS" "OPTIONAL MATCH (u:User {userId:'probe'}) OPTIONAL MATCH (a:Agent {agentId:'probe'}) WITH u, a, (u IS NULL AND a IS NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (:User {userId:'probe', displayName:'p', email:'p'})) RETURN ok, u IS NOT NULL, a IS NOT NULL")
assert_index_scan "guarded ensure existence checks" "$prof"
gq "$WS" "MATCH (u:User {userId:'probe'}) DELETE u" > /dev/null

out=$(gq "$WS" "MATCH (a:Agent {agentId:'bot1'}) MATCH (c:Channel {channelId:'ch1'}) MERGE (a)-[r:MEMBER_OF]->(c) ON CREATE SET r.role='assistant', r.joinedAt=1000 RETURN r.role")
assert_contains "add agent to channel" "assistant" "$out"

out=$(gq "$WS" "MATCH (u)-[:MEMBER_OF]->(c:Channel {channelId:'ch1'}) RETURN coalesce(u.userId, u.agentId) AS memberId, u.displayName AS displayName, labels(u) AS type ORDER BY u.displayName")
assert_contains "channel members: Alice" "u1" "$out"
assert_contains "channel members: Bob" "u2" "$out"
assert_contains "channel members: bot1" "bot1" "$out"
assert_contains "channel members: Agent label visible" "Agent" "$out"

# agent posts a message (assistant role) — v2 path resolves Agent authors via agentId index
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'m5'}) OPTIONAL MATCH (ua:User {userId:'bot1'}) OPTIONAL MATCH (aa:Agent {agentId:'bot1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'m5', text:'I am the AI assistant', role:'assistant', createdAt:3000, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=3000 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "agent posts assistant message (v2 write commits)" "w=true auth=true" "$out"

# ── §9: mentions & read-cursors (MCP transport) ──────────────────────────────

echo ""
echo "▶ §9 mentions — write path (MENTIONS_MEMBER)"

# first message in th2 with EMPTY mentions — the CASE guard is load-bearing for the
# write itself in v2: a bare UNWIND [] would collapse the stream before the FOREACH.
out=$(gq "$WS" "CYPHER mentions=[] MATCH (t:Thread {threadId:'th2'}) OPTIONAL MATCH (t)-[:HEAD]->(h) OPTIONAL MATCH (dup:Message {msgId:'mn1'}) OPTIONAL MATCH (ua:User {userId:'u1'}) OPTIONAL MATCH (aa:Agent {agentId:'u1'}) WITH t, h, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, h, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, h, dup, author, mems, (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'mn1', text:'mentions probe first', role:'user', createdAt:4000, threadId:'th2'}) CREATE (t)-[:HEAD]->(m) CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=4000 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok) AS status")
assert_contains "mentions=[] v2 first-post commits (guard keeps the write alive)" "w=true" "$out"

out=$(gq "$WS" "MATCH (:Message {msgId:'mn1'})-[r:MENTIONS_MEMBER]->() RETURN count(r) AS n")
assert_contains "mentions=[] creates zero MENTIONS_MEMBER edges" "0" "$out"

# subsequent message mentioning u2 AND agent bot1 (mixed User/Agent resolution)
out=$(gq "$WS" "CYPHER mentions=['u2','bot1'] MATCH (t:Thread {threadId:'th2'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'mn2'}) OPTIONAL MATCH (ua:User {userId:'u1'}) OPTIONAL MATCH (aa:Agent {agentId:'u1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'mn2', text:'ping u2 and bot1', role:'user', createdAt:4001, threadId:'th2'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=4001 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok) AS status")
assert_contains "mention post mn2 commits (v2)" "w=true" "$out"

out=$(gq "$WS" "MATCH (:Message {msgId:'mn2'})-[:MENTIONS_MEMBER]->(x) RETURN count(x) AS n, collect(coalesce(x.userId,x.agentId)) AS who")
assert_contains "mn2 has exactly 2 MENTIONS_MEMBER edges" "2" "$out"
assert_contains "mn2 mentions User u2 (resolved via User index)" "u2" "$out"
assert_contains "mn2 mentions Agent bot1 (resolved via Agent index)" "bot1" "$out"

# dedup + unknown-skip: ['u1','u1','nope'] → one edge to u1, 'nope' dropped
out=$(gq "$WS" "CYPHER mentions=['u1','u1','nope'] MATCH (t:Thread {threadId:'th2'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'mn3'}) OPTIONAL MATCH (ua:User {userId:'u2'}) OPTIONAL MATCH (aa:Agent {agentId:'u2'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems WITH t, tailRel, prev, dup, author, mems, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'mn3', text:'dup and unknown mentions', role:'user', createdAt:4002, threadId:'th2'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=4002 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))) RETURN 'w='+toString(ok) AS status")
assert_contains "dedup post mn3 commits (v2)" "w=true" "$out"

out=$(gq "$WS" "MATCH (:Message {msgId:'mn3'})-[:MENTIONS_MEMBER]->(x) RETURN count(x) AS n, collect(coalesce(x.userId,x.agentId)) AS who")
assert_contains "mn3 dedups duplicate mention to a single edge" "1" "$out"
assert_not_contains "mn3 drops unknown mention 'nope'" "nope" "$out"

echo ""
echo "▶ §9.1/§9.2 since-reads with mention flag (chronological, (createdAt,msgId) order)"

# §9.1 thread-scoped, reader = bot1, since 3999 → mn1,mn2,mn3 chronological; mn2 (mentions bot1) flagged
out=$(rq "$WS" "CYPHER threadId='th2' since=3999 sinceMsgId='' meId='bot1' limit=50 MATCH (t:Thread {threadId:\$threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.createdAt > \$since OR (m.createdAt = \$since AND m.msgId > \$sinceMsgId) MATCH (m)-[:POSTED_BY]->(author) OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) WHERE me.userId=\$meId OR me.agentId=\$meId WITH m, author, count(me) > 0 AS isMention RETURN m.msgId, coalesce(author.userId,author.agentId) AS authorId, isMention ORDER BY m.createdAt, m.msgId LIMIT \$limit")
assert_contains "§9.1 returns mn1 (after since)" "mn1" "$out"
assert_contains "§9.1 returns mn2" "mn2" "$out"
# chronological order — the cursor-pagination invariant: a LIMIT-truncated page
# must be the earliest rows (mention-first sorting + LIMIT loses messages);
# mn2's mention of the reader is carried by the isMention flag instead
mn1_line=$(echo "$out" | grep -n "mn1" | head -1 | cut -d: -f1)
mn2_line=$(echo "$out" | grep -n "mn2" | head -1 | cut -d: -f1)
if [ -n "$mn1_line" ] && [ -n "$mn2_line" ] && [ "$mn1_line" -lt "$mn2_line" ]; then
  echo "  ✓ §9.1 chronological order (mn1 before mn2; mention flagged, not resorted)"
  PASS=$((PASS+1))
else
  echo "  ✗ §9.1 chronological ordering wrong (mn1 line ${mn1_line}, mn2 line ${mn2_line})"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi

# §9.1 explicit since keeps plain-`>` semantics — excludes earlier messages
out=$(rq "$WS" "CYPHER threadId='th2' since=4001 meId='bot1' limit=50 MATCH (t:Thread {threadId:\$threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.createdAt > \$since MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId ORDER BY m.createdAt, m.msgId LIMIT \$limit")
assert_contains     "§9.1 since=4001 includes mn3" "mn3" "$out"
assert_not_contains "§9.1 since=4001 excludes mn1" "mn1" "$out"
assert_not_contains "§9.1 since=4001 excludes mn2" "mn2" "$out"

# §9.2 workspace-wide since read (composite keyset) must stay an index scan on Message.createdAt
prof=$(gp "$WS" "CYPHER since=3999 sinceMsgId='' meId='bot1' limit=50 MATCH (m:Message) WHERE m.createdAt > \$since OR (m.createdAt = \$since AND m.msgId > \$sinceMsgId) MATCH (m)-[:POSTED_BY]->(author) OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me) WHERE me.userId=\$meId OR me.agentId=\$meId WITH m, author, count(me) > 0 AS isMention RETURN m.msgId, isMention ORDER BY m.createdAt, m.msgId LIMIT \$limit")
assert_index_scan "§9.2 composite keyset since-read uses Message.createdAt index" "$prof"

echo ""
echo "▶ §9 keyset paging across a millisecond tie (K-007 tie-skip regression)"

# seed th3: k1@5000, k2@5001, k3@5001 (tie), k4@5002 — structural seed of a v2-shaped chain
gq "$WS" "MATCH (c:Channel {channelId:'ch1'}) MATCH (u:User {userId:'u1'}) CREATE (t:Thread {threadId:'th3', title:'tie probe', createdAt:5000, updatedAt:5002}) CREATE (c)-[:HAS_THREAD]->(t) CREATE (k1:Message {msgId:'k1', text:'tie one', role:'user', createdAt:5000, threadId:'th3'}) CREATE (k2:Message {msgId:'k2', text:'tie two', role:'user', createdAt:5001, threadId:'th3'}) CREATE (k3:Message {msgId:'k3', text:'tie three', role:'user', createdAt:5001, threadId:'th3'}) CREATE (k4:Message {msgId:'k4', text:'tie four', role:'user', createdAt:5002, threadId:'th3'}) CREATE (t)-[:HEAD]->(k1) CREATE (t)-[:TAIL]->(k4) CREATE (k1)-[:NEXT]->(k2) CREATE (k2)-[:NEXT]->(k3) CREATE (k3)-[:NEXT]->(k4) CREATE (k1)-[:POSTED_BY]->(u) CREATE (k2)-[:POSTED_BY]->(u) CREATE (k3)-[:POSTED_BY]->(u) CREATE (k4)-[:POSTED_BY]->(u)" > /dev/null

# page 1 (limit 2) ends exactly on the tie boundary: k1,k2
out=$(rq "$WS" "CYPHER threadId='th3' since=0 sinceMsgId='' meId='u1' limit=2 MATCH (t:Thread {threadId:\$threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.createdAt > \$since OR (m.createdAt = \$since AND m.msgId > \$sinceMsgId) MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId ORDER BY m.createdAt, m.msgId LIMIT \$limit")
if echo "$out" | grep -qF "k2" && ! echo "$out" | grep -qF "k3"; then
  echo "  ✓ §9 keyset page 1 (limit 2) ends at the tie boundary (k1,k2)"
  PASS=$((PASS+1))
else
  echo "  ✗ §9 keyset page 1 wrong"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi

# page 2 resumes from the composite pair (5001,'k2') → the tied sibling k3 is delivered
out=$(rq "$WS" "CYPHER threadId='th3' since=5001 sinceMsgId='k2' meId='u1' limit=50 MATCH (t:Thread {threadId:\$threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.createdAt > \$since OR (m.createdAt = \$since AND m.msgId > \$sinceMsgId) MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId ORDER BY m.createdAt, m.msgId LIMIT \$limit")
assert_contains "§9 keyset page 2 delivers the tied sibling k3 (tie-skip regression)" "k3" "$out"
if echo "$out" | grep -qF "k4" && ! echo "$out" | grep -qF "k2"; then
  echo "  ✓ §9 keyset page 2 completes the set (k4 in, boundary k2 excluded — nothing skipped)"
  PASS=$((PASS+1))
else
  echo "  ✗ §9 keyset page 2 wrong (expected k3,k4 only)"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi

# §9.2 RETURN carries threadId (denormalized navigation metadata)
out=$(rq "$WS" "CYPHER since=4999 sinceMsgId='' meId='bot1' limit=50 MATCH (m:Message) WHERE m.createdAt > \$since OR (m.createdAt = \$since AND m.msgId > \$sinceMsgId) MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId, m.threadId AS threadId ORDER BY m.createdAt, m.msgId LIMIT \$limit")
assert_contains "§9.2 rows carry threadId" "th3" "$out"

echo ""
echo "▶ §9.3/§9.4 read-cursors (composite monotonic)"

# §9.3 v2 create: advance bot1:th2 to (4001,'mn2')
out=$(gq "$WS" "CYPHER meId='bot1' threadId='th2' cursorId='bot1:th2' now=4001 nowMsgId='mn2' MATCH (mem) WHERE mem.userId=\$meId OR mem.agentId=\$meId MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:\$cursorId}) ON CREATE SET rc.memberId=\$meId, rc.threadId=\$threadId WITH rc, (\$now > coalesce(rc.lastReadAt,0) OR (\$now = coalesce(rc.lastReadAt,0) AND \$nowMsgId > coalesce(rc.lastReadMsgId,''))) AS adv SET rc.lastReadAt = CASE WHEN adv THEN \$now ELSE rc.lastReadAt END, rc.lastReadMsgId = CASE WHEN adv THEN \$nowMsgId ELSE rc.lastReadMsgId END RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.3 cursor advances to (4001,'mn2')" "mn2" "$out"

# tie, larger msgId → advances to (4001,'mn3')
out=$(gq "$WS" "CYPHER meId='bot1' threadId='th2' cursorId='bot1:th2' now=4001 nowMsgId='mn3' MATCH (mem) WHERE mem.userId=\$meId OR mem.agentId=\$meId MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:\$cursorId}) ON CREATE SET rc.memberId=\$meId, rc.threadId=\$threadId WITH rc, (\$now > coalesce(rc.lastReadAt,0) OR (\$now = coalesce(rc.lastReadAt,0) AND \$nowMsgId > coalesce(rc.lastReadMsgId,''))) AS adv SET rc.lastReadAt = CASE WHEN adv THEN \$now ELSE rc.lastReadAt END, rc.lastReadMsgId = CASE WHEN adv THEN \$nowMsgId ELSE rc.lastReadMsgId END RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.3 tie with larger msgId advances to 'mn3'" "mn3" "$out"

# tie, smaller msgId (stale replay) → refused, stays 'mn3'
out=$(gq "$WS" "CYPHER meId='bot1' threadId='th2' cursorId='bot1:th2' now=4001 nowMsgId='mn2' MATCH (mem) WHERE mem.userId=\$meId OR mem.agentId=\$meId MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:\$cursorId}) ON CREATE SET rc.memberId=\$meId, rc.threadId=\$threadId WITH rc, (\$now > coalesce(rc.lastReadAt,0) OR (\$now = coalesce(rc.lastReadAt,0) AND \$nowMsgId > coalesce(rc.lastReadMsgId,''))) AS adv SET rc.lastReadAt = CASE WHEN adv THEN \$now ELSE rc.lastReadAt END, rc.lastReadMsgId = CASE WHEN adv THEN \$nowMsgId ELSE rc.lastReadMsgId END RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.3 tie with smaller msgId refused (stays 'mn3')" "mn3" "$out"

# backward timestamp → refused (stays 4001)
out=$(gq "$WS" "CYPHER meId='bot1' threadId='th2' cursorId='bot1:th2' now=4000 nowMsgId='zz9' MATCH (mem) WHERE mem.userId=\$meId OR mem.agentId=\$meId MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:\$cursorId}) ON CREATE SET rc.memberId=\$meId, rc.threadId=\$threadId WITH rc, (\$now > coalesce(rc.lastReadAt,0) OR (\$now = coalesce(rc.lastReadAt,0) AND \$nowMsgId > coalesce(rc.lastReadMsgId,''))) AS adv SET rc.lastReadAt = CASE WHEN adv THEN \$now ELSE rc.lastReadAt END, rc.lastReadMsgId = CASE WHEN adv THEN \$nowMsgId ELSE rc.lastReadMsgId END RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.3 backward advance is a no-op (stays 4001)" "4001" "$out"

# forward → advances to (4002,'mn9')
out=$(gq "$WS" "CYPHER meId='bot1' threadId='th2' cursorId='bot1:th2' now=4002 nowMsgId='mn9' MATCH (mem) WHERE mem.userId=\$meId OR mem.agentId=\$meId MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:\$cursorId}) ON CREATE SET rc.memberId=\$meId, rc.threadId=\$threadId WITH rc, (\$now > coalesce(rc.lastReadAt,0) OR (\$now = coalesce(rc.lastReadAt,0) AND \$nowMsgId > coalesce(rc.lastReadMsgId,''))) AS adv SET rc.lastReadAt = CASE WHEN adv THEN \$now ELSE rc.lastReadAt END, rc.lastReadMsgId = CASE WHEN adv THEN \$nowMsgId ELSE rc.lastReadMsgId END RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.3 forward advance moves cursor to 4002" "4002" "$out"

# MERGE is idempotent — exactly one ReadCursor for this (member, thread)
out=$(gq "$WS" "MATCH (rc:ReadCursor {cursorId:'bot1:th2'}) RETURN count(rc) AS n")
assert_contains "§9.3 exactly one ReadCursor node (MERGE idempotent)" "1" "$out"

# ReadCursor uniqueness constraint blocks a duplicate cursorId
out=$(gq "$WS" "CREATE (:ReadCursor {cursorId:'bot1:th2'})" 2>&1)
assert_contains "ReadCursor constraint blocks duplicate cursorId" "unique constraint violation" "$out"

# §9.4 read the cursor back (point lookup on cursorId) — returns the composite pair
out=$(rq "$WS" "CYPHER cursorId='bot1:th2' MATCH (rc:ReadCursor {cursorId:\$cursorId}) RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_contains "§9.4 reads back cursor lastReadAt" "4002" "$out"
assert_contains "§9.4 reads back cursor lastReadMsgId (composite pair)" "mn9" "$out"

prof=$(gp "$WS" "MATCH (rc:ReadCursor {cursorId:'bot1:th2'}) RETURN rc.lastReadAt, rc.lastReadMsgId")
assert_index_scan "§9.4 cursor read uses ReadCursor.cursorId index" "$prof"

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

prof=$(gp "$WS" "MATCH (c:Channel) WHERE c.channelId > '' RETURN c.channelId, c.name, c.createdAt ORDER BY c.createdAt DESC LIMIT 100")
assert_index_scan "list channels uses index" "$prof"

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

# ── §EMITTED: agent answer provenance (K-013) ────────────────────────────────
#
# The AI responder posts its answer as the Agent (role derived = assistant) via the
# §4 subsequent write path, and in the SAME GRAPH.QUERY creates
# (answer)-[:EMITTED {score, rank}]->(seed) provenance edges to the retrieval seeds.
# EMITTED rides inside the guarded FOREACH exactly like MENTIONS_MEMBER: same
# empty-UNWIND CASE guard, so $seedIds=[] is a true no-op and a dupMsg replay writes
# provenance exactly once. Per-edge score/rank come from map params keyed by the
# seed's msgId — a map-projection cannot be a CREATE endpoint on this build, so the
# seed must be a bound node var (collect(DISTINCT s)); the props are looked up via
# $scoreBy[seed.msgId] / $rankBy[seed.msgId]. Canonical bodies: QUERIES.md §4/§10.

echo ""
echo "▶ §EMITTED agent answer provenance (K-013)"

# agent bot1 answers into th1 (subsequent path; TAIL=m5), mentions u1,
# cites m1,m2 as provenance (+ unknown 'nope' which must be skipped)
out=$(gq "$WS" "CYPHER mentions=['u1'] seedIds=['m1','m2','nope'] scoreBy={m1:0.0,m2:0.006} rankBy={m1:0,m2:1} MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'ag1'}) OPTIONAL MATCH (ua:User {userId:'bot1'}) OPTIONAL MATCH (aa:Agent {agentId:'bot1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems UNWIND (CASE WHEN \$seedIds = [] THEN [null] ELSE \$seedIds END) AS sid OPTIONAL MATCH (s:Message {msgId: sid}) WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT s) AS seeds WITH t, tailRel, prev, dup, author, mems, seeds, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'ag1', text:'grounded answer', role:'assistant', createdAt:6000, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=6000 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {score: \$scoreBy[seed.msgId], rank: \$rankBy[seed.msgId]}]->(seed))) RETURN 'w='+toString(ok)+' auth='+toString(author IS NOT NULL) AS status")
assert_contains "§EMITTED agent answer ag1 commits (agent-authored)" "w=true auth=true" "$out"

# K-007 invariant: agent answer reads role assistant, author resolves to an Agent
out=$(rq "$WS" "MATCH (m:Message {msgId:'ag1'})-[:POSTED_BY]->(a) RETURN m.role, labels(a) AS authorType")
assert_contains "§EMITTED ag1 role derived assistant (K-007 invariant)" "assistant" "$out"
assert_contains "§EMITTED ag1 author resolves to Agent" "Agent" "$out"

# provenance read: given the answer, what did it cite (ordered by rank)?
out=$(rq "$WS" "MATCH (a:Message {msgId:'ag1'})-[e:EMITTED]->(s:Message) RETURN s.msgId, s.text, e.score, e.rank ORDER BY e.rank")
assert_contains "§EMITTED provenance read cites m1" "m1" "$out"
assert_contains "§EMITTED provenance read cites m2" "m2" "$out"

# unknown seed 'nope' skipped → exactly 2 provenance edges (collect drops the null)
out=$(rq "$WS" "MATCH (:Message {msgId:'ag1'})-[e:EMITTED]->() RETURN count(e) AS n")
assert_contains "§EMITTED ag1 has exactly 2 provenance edges (unknown seed skipped)" "2" "$out"

# each EMITTED edge carries score + rank; the top seed m1 is rank 0
out=$(rq "$WS" "MATCH (:Message {msgId:'ag1'})-[e:EMITTED]->(:Message {msgId:'m1'}) RETURN 'rank='+toString(e.rank)+' hasScore='+toString(e.score IS NOT NULL) AS p")
assert_contains "§EMITTED edge carries rank + score props" "rank=0 hasScore=true" "$out"

# reverse read: given a seed, which answers cited it?
out=$(rq "$WS" "MATCH (a:Message)-[e:EMITTED]->(:Message {msgId:'m1'}) RETURN a.msgId, e.rank")
assert_contains "§EMITTED reverse read: m1 cited by ag1" "ag1" "$out"

# empty seedIds=[] is a true no-op — a non-answer message still commits with ZERO
# EMITTED edges (the CASE guard keeps the write itself alive)
out=$(gq "$WS" "CYPHER mentions=[] seedIds=[] scoreBy={} rankBy={} MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'ag2'}) OPTIONAL MATCH (ua:User {userId:'bot1'}) OPTIONAL MATCH (aa:Agent {agentId:'bot1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems UNWIND (CASE WHEN \$seedIds = [] THEN [null] ELSE \$seedIds END) AS sid OPTIONAL MATCH (s:Message {msgId: sid}) WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT s) AS seeds WITH t, tailRel, prev, dup, author, mems, seeds, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'ag2', text:'no provenance msg', role:'assistant', createdAt:6001, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=6001 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {score: \$scoreBy[seed.msgId], rank: \$rankBy[seed.msgId]}]->(seed))) RETURN 'w='+toString(ok) AS status")
assert_contains "§EMITTED empty seedIds=[] still commits the write" "w=true" "$out"
out=$(rq "$WS" "MATCH (:Message {msgId:'ag2'})-[e:EMITTED]->() RETURN count(e) AS n")
assert_contains "§EMITTED empty seedIds=[] creates zero provenance edges" "0" "$out"

# dupMsg replay of ag1 → whole write no-ops; provenance stays exactly 2 (exactly-once)
out=$(gq "$WS" "CYPHER mentions=['u1'] seedIds=['m1','m2'] scoreBy={m1:0.0,m2:0.006} rankBy={m1:0,m2:1} MATCH (t:Thread {threadId:'th1'})-[tailRel:TAIL]->(prev:Message) OPTIONAL MATCH (dup:Message {msgId:'ag1'}) OPTIONAL MATCH (ua:User {userId:'bot1'}) OPTIONAL MATCH (aa:Agent {agentId:'bot1'}) WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author UNWIND (CASE WHEN \$mentions = [] THEN [null] ELSE \$mentions END) AS mid OPTIONAL MATCH (mu:User {userId: mid}) OPTIONAL MATCH (ma:Agent {agentId: mid}) WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems UNWIND (CASE WHEN \$seedIds = [] THEN [null] ELSE \$seedIds END) AS sid OPTIONAL MATCH (s:Message {msgId: sid}) WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT s) AS seeds WITH t, tailRel, prev, dup, author, mems, seeds, (dup IS NULL AND author IS NOT NULL) AS ok FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (m:Message {msgId:'ag1', text:'grounded answer', role:'assistant', createdAt:6000, threadId:'th1'}) CREATE (prev)-[:NEXT]->(m) DELETE tailRel CREATE (t)-[:TAIL]->(m) CREATE (m)-[:POSTED_BY]->(author) SET t.updatedAt=6000 FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem)) FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {score: \$scoreBy[seed.msgId], rank: \$rankBy[seed.msgId]}]->(seed))) RETURN 'w='+toString(ok)+' dup='+toString(dup IS NOT NULL) AS status")
assert_contains "§EMITTED dupMsg replay no-ops (idempotent)" "w=false dup=true" "$out"
out=$(rq "$WS" "MATCH (:Message {msgId:'ag1'})-[e:EMITTED]->() RETURN count(e) AS n")
assert_contains "§EMITTED provenance written exactly once (still 2 after replay)" "2" "$out"

# provenance read anchors on the Message.msgId index (no label scan)
prof=$(gp "$WS" "MATCH (a:Message {msgId:'ag1'})-[e:EMITTED]->(s:Message) RETURN s.msgId, e.score, e.rank ORDER BY e.rank")
assert_index_scan "§EMITTED provenance read uses Message.msgId index" "$prof"

# ── §11: workflow definitions & snapshots (M3 Slice 1) ───────────────────────
#
# Canonical bodies: QUERIES.md §11. Defs live in `reference`; publishing then
# materializing into ws:{id} are the two write paths. Step identity is the
# synthetic stepUid = "{defKey}:{version}:{stepKey}" (index + UNIQUE, both graphs);
# HAS_STEP gives the def/snapshot an index-anchored handle on all its steps (the
# §B8 STARTS-WITH-degrades-to-label-scan resolution). config/guard are opaque.
# Distinct def keys (wf_review / wf_triage) keep these independent of the §ref
# WorkflowDef composite-constraint block above.

# publish query (reference) — the K-020 write path, verbatim from QUERIES.md §11.1
PUBLISH='CYPHER key="wf_review" version="1" name="Review" kind="process" startKey="start" steps=[{key:"start",type:"message",config:"{}"},{key:"gather",type:"human",config:"{\"form\":\"x\"}"},{key:"decide",type:"decision",config:"{}"},{key:"done",type:"message",config:"{}"}] transitions=[{from:"start",to:"gather",on:"submit",guard:"",order:0},{from:"gather",to:"decide",on:"submit",guard:"",order:0},{from:"decide",to:"done",on:"approve",guard:"ctx.ok",order:0},{from:"decide",to:"gather",on:"reject",guard:"",order:1}]
MERGE (d:WorkflowDef {key: $key, version: $version}) ON CREATE SET d.name = $name, d.kind = $kind
WITH d UNWIND $steps AS s MERGE (st:Step {stepUid: $key + ":" + $version + ":" + s.key}) ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config MERGE (d)-[:HAS_STEP]->(st)
WITH d, count(st) AS stepCount MATCH (start:Step {stepUid: $key + ":" + $version + ":" + $startKey}) MERGE (d)-[:START]->(start)
WITH d, stepCount UNWIND $transitions AS tr MATCH (from:Step {stepUid: $key + ":" + $version + ":" + tr.from}) MATCH (to:Step {stepUid: $key + ":" + $version + ":" + tr.to}) MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to) ON CREATE SET rel.guard = tr.guard
WITH d, stepCount, count(rel) AS transitionCount RETURN d.key AS key, d.version AS version, stepCount, transitionCount'

echo ""
echo "▶ §11 workflow definitions (reference — K-020)"

# publish authors the full subgraph: 5 nodes (def + 4 steps), 9 rels (4 HAS_STEP + START + 4 TRANSITION)
out=$(gq "$REF" "$PUBLISH")
assert_contains "§11.1 publish wf_review v1 authors 5 nodes"       "Nodes created: 5"          "$out"
assert_contains "§11.1 publish wf_review v1 authors 9 rels"        "Relationships created: 9"  "$out"

out=$(gq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(s:Step) RETURN count(s) AS n")
assert_contains "§11.1 publish creates 4 steps (HAS_STEP)" "4" "$out"
out=$(gq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:START]->(s:Step) RETURN count(s) AS n, collect(s.key) AS startKey")
assert_contains "§11.1 publish creates exactly one START edge" "1" "$out"
assert_contains "§11.1 START points at 'start' step" "start" "$out"
out=$(gq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(:Step)-[r:TRANSITION]->(:Step) RETURN count(r) AS n")
assert_contains "§11.1 publish creates 4 TRANSITION edges" "4" "$out"

# stepUid UNIQUE constraint (reference) blocks a duplicate step identity
out=$(gq "$REF" "CREATE (:Step {stepUid:'wf_review:1:start', key:'start', type:'message', config:'{}'})" 2>&1)
assert_contains "§11 Step.stepUid constraint blocks duplicate (reference)" "unique constraint violation" "$out"

# idempotent re-publish is a structural no-op (0 nodes/rels created)
out=$(gq "$REF" "$PUBLISH")
assert_not_contains "§11.1 idempotent re-publish creates no nodes" "Nodes created" "$out"
assert_not_contains "§11.1 idempotent re-publish creates no rels"  "Relationships created" "$out"
out=$(gq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(s:Step) RETURN count(s) AS n")
assert_contains "§11.1 re-publish left exactly 4 steps" "4" "$out"

# read-def subgraph (§11.2): meta + steps, then transitions
out=$(rq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'}) OPTIONAL MATCH (d)-[:START]->(start:Step) OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step) RETURN d.name AS name, d.kind AS kind, start.key AS startKey, collect(DISTINCT {key:s.key, type:s.type, config:s.config}) AS steps")
assert_contains "§11.2a read-def returns startKey 'start'" "start" "$out"
assert_contains "§11.2a read-def returns kind 'process'"   "process" "$out"
assert_contains "§11.2a read-def carries a step (gather)"   "gather" "$out"
out=$(rq "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step) RETURN collect({from:from.key, to:to.key, on:tr.on, guard:tr.guard, order:tr.order}) AS transitions")
assert_contains "§11.2b read-def transitions include 'approve'" "approve" "$out"
assert_contains "§11.2b read-def transitions carry guard verbatim (ctx.ok)" "ctx.ok" "$out"

# publish a v2 (meta-only MERGE) so get-latest has something to rank
gq "$REF" 'CYPHER key="wf_review" version="2" name="Review v2" kind="process" MERGE (d:WorkflowDef {key:$key,version:$version}) ON CREATE SET d.name=$name, d.kind=$kind RETURN d.version' > /dev/null
# publish a second def key for the list test
gq "$REF" 'CYPHER key="wf_triage" version="1" name="Triage" kind="conversation" MERGE (d:WorkflowDef {key:$key,version:$version}) ON CREATE SET d.name=$name, d.kind=$kind RETURN d.version' > /dev/null

out=$(rq "$REF" "MATCH (d:WorkflowDef {key:'wf_review'}) RETURN d.key, d.version, d.name ORDER BY d.version DESC LIMIT 1")
assert_contains "§11.3 get-def latest returns version 2" "2" "$out"
assert_contains "§11.3 get-def latest returns 'Review v2'" "Review v2" "$out"
out=$(rq "$REF" "MATCH (d:WorkflowDef {key:'wf_review', version:'1'}) RETURN d.name, d.kind")
assert_contains "§11.3 get-def specific version 1 returns 'Review'" "Review" "$out"
out=$(rq "$REF" "CYPHER limit=50 MATCH (d:WorkflowDef) WHERE d.key > '' RETURN d.key, d.version, d.name, d.kind ORDER BY d.key, d.version DESC LIMIT \$limit")
assert_contains "§11.3 list-defs includes wf_review" "wf_review" "$out"
assert_contains "§11.3 list-defs includes wf_triage" "wf_triage" "$out"

# PROFILE: reads anchor on WorkflowDef.key index, no label scan
prof=$(gp "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(s:Step) RETURN s.key, s.type")
assert_index_scan "§11.2a read-def steps anchors on WorkflowDef index" "$prof"
prof=$(gp "$REF" "MATCH (d:WorkflowDef {key:'wf_review',version:'1'})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step) RETURN from.key, to.key, tr.on")
assert_index_scan "§11.2b read-def transitions anchors on WorkflowDef index" "$prof"
prof=$(gp "$REF" "CYPHER limit=50 MATCH (d:WorkflowDef) WHERE d.key > '' RETURN d.key, d.version ORDER BY d.key, d.version DESC LIMIT \$limit")
assert_index_scan "§11.3 list-defs uses WorkflowDef.key index" "$prof"

# ── §11 materialization (workspace — K-021) ──────────────────────────────────
#
# Same shape scoped to ws:{id} with the WorkflowDefSnapshot root label. In the app
# this is two-phase (read reference §11.2 → write workspace §11.4); the test drives
# the write directly with the def structure as params.

MATERIALIZE='CYPHER key="wf_review" version="1" name="Review" kind="process" startKey="start" steps=[{key:"start",type:"message",config:"{}"},{key:"gather",type:"human",config:"{\"form\":\"x\"}"},{key:"decide",type:"decision",config:"{}"},{key:"done",type:"message",config:"{}"}] transitions=[{from:"start",to:"gather",on:"submit",guard:"",order:0},{from:"gather",to:"decide",on:"submit",guard:"",order:0},{from:"decide",to:"done",on:"approve",guard:"ctx.ok",order:0},{from:"decide",to:"gather",on:"reject",guard:"",order:1}]
MERGE (snap:WorkflowDefSnapshot {key: $key, version: $version}) ON CREATE SET snap.name = $name, snap.kind = $kind
WITH snap UNWIND $steps AS s MERGE (st:Step {stepUid: $key + ":" + $version + ":" + s.key}) ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config MERGE (snap)-[:HAS_STEP]->(st)
WITH snap, count(st) AS stepCount MATCH (start:Step {stepUid: $key + ":" + $version + ":" + $startKey}) MERGE (snap)-[:START]->(start)
WITH snap, stepCount UNWIND $transitions AS tr MATCH (from:Step {stepUid: $key + ":" + $version + ":" + tr.from}) MATCH (to:Step {stepUid: $key + ":" + $version + ":" + tr.to}) MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to) ON CREATE SET rel.guard = tr.guard
WITH snap, stepCount, count(rel) AS transitionCount RETURN snap.key AS key, snap.version AS version, stepCount, transitionCount'

echo ""
echo "▶ §11 snapshot materialization (workspace — K-021)"

out=$(gq "$WS" "$MATERIALIZE")
assert_contains "§11.4 materialize wf_review v1 authors 5 local nodes" "Nodes created: 5"         "$out"
assert_contains "§11.4 materialize wf_review v1 authors 9 local rels"  "Relationships created: 9" "$out"

out=$(gq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:HAS_STEP]->(s:Step) RETURN count(s) AS n")
assert_contains "§11.4 materialize creates 4 local steps" "4" "$out"
out=$(gq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:START]->(s:Step) RETURN count(s) AS n")
assert_contains "§11.4 materialize creates one local START" "1" "$out"
out=$(gq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:HAS_STEP]->(:Step)-[r:TRANSITION]->(:Step) RETURN count(r) AS n")
assert_contains "§11.4 materialize creates 4 local TRANSITION edges" "4" "$out"

# workspace composite {key,version} constraint (not asserted before this slice)
out=$(gq "$WS" "CREATE (:WorkflowDefSnapshot {key:'wf_review', version:'1', name:'dupe'})" 2>&1)
assert_contains "§11 WorkflowDefSnapshot composite constraint blocks dup {key,version}" "unique constraint violation" "$out"
# workspace Step.stepUid constraint
out=$(gq "$WS" "CREATE (:Step {stepUid:'wf_review:1:start', key:'start', type:'message', config:'{}'})" 2>&1)
assert_contains "§11 Step.stepUid constraint blocks duplicate (workspace)" "unique constraint violation" "$out"

# read-snapshot subgraph (§11.5)
out=$(rq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'}) OPTIONAL MATCH (snap)-[:START]->(start:Step) OPTIONAL MATCH (snap)-[:HAS_STEP]->(s:Step) RETURN snap.name AS name, snap.kind AS kind, start.key AS startKey, collect(DISTINCT {key:s.key, type:s.type, config:s.config}) AS steps")
assert_contains "§11.5 read-snapshot returns startKey 'start'" "start" "$out"
assert_contains "§11.5 read-snapshot carries a step (decide)"  "decide" "$out"
out=$(rq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step) RETURN collect({from:from.key, to:to.key, on:tr.on, guard:tr.guard, order:tr.order}) AS transitions")
assert_contains "§11.5 read-snapshot transitions include 'approve'" "approve" "$out"

# list / get snapshot (§11.6)
out=$(rq "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review', version:'1'}) RETURN snap.name, snap.kind")
assert_contains "§11.6 get-snapshot specific version returns 'Review'" "Review" "$out"
out=$(rq "$WS" "CYPHER limit=50 MATCH (snap:WorkflowDefSnapshot) WHERE snap.key > '' RETURN snap.key, snap.version, snap.name ORDER BY snap.key, snap.version DESC LIMIT \$limit")
assert_contains "§11.6 list-snapshots includes wf_review" "wf_review" "$out"

# idempotent re-materialize is a structural no-op
out=$(gq "$WS" "$MATERIALIZE")
assert_not_contains "§11.4 idempotent re-materialize creates no nodes" "Nodes created" "$out"
assert_not_contains "§11.4 idempotent re-materialize creates no rels"  "Relationships created" "$out"

# PROFILE: snapshot reads anchor on WorkflowDefSnapshot.key index, no label scan
prof=$(gp "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:HAS_STEP]->(s:Step) RETURN s.key, s.type")
assert_index_scan "§11.5 read-snapshot steps anchors on WorkflowDefSnapshot index" "$prof"
prof=$(gp "$WS" "MATCH (snap:WorkflowDefSnapshot {key:'wf_review',version:'1'})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step) RETURN from.key, to.key, tr.on")
assert_index_scan "§11.5 read-snapshot transitions anchors on WorkflowDefSnapshot index" "$prof"

# ── §12: workflow execution — runs, step-runs & traces (M3 executor — K-022) ──
#
# Canonical bodies: QUERIES.md §12 (1:1 with repository.py, U3). Runs against the
# wf_review snapshot materialized into ws:test by the §11 block above (steps
# start→gather→decide→done). Uses th1 + m1 as the run's thread + trigger message.
# Every state-move is a single GRAPH.QUERY; each read anchors on an index (asserted
# via PROFILE). PRODUCED (StepRun→Message) is DISTINCT from EMITTED (Message→Message,
# §10) — D2. record_step_and_advance is the M4 tail-anchored O(1) atomic advance.

echo ""
echo "▶ §12 workflow execution — start_run"

# 12.1 start_run: create wr1 OF_DEF wf_review, AT_STEP start, TRIGGERED_BY m1
out=$(gq "$WS" 'CYPHER runId="wr1" defKey="wf_review" defVersion="1" startedAt=7000 ctx="{}" trace=true maxSteps=12 triggerMsgId="m1" MATCH (snap:WorkflowDefSnapshot {key:$defKey, version:$defVersion})-[:START]->(start:Step) MATCH (trigger:Message {msgId:$triggerMsgId}) CREATE (r:WorkflowRun {runId:$runId, defKey:$defKey, defVersion:$defVersion, status:"running", startedAt:$startedAt, ctx:$ctx, trace:$trace, maxSteps:$maxSteps, stepCount:0, waitingThreadId:""}) CREATE (r)-[:OF_DEF]->(snap) CREATE (r)-[:AT_STEP]->(start) CREATE (r)-[:TRIGGERED_BY]->(trigger) RETURN "start="+start.key+" status="+r.status+" sc="+toString(r.stepCount) AS s')
assert_contains "§12.1 start_run creates wr1 at START step, running, stepCount 0" "start=start status=running sc=0" "$out"

# start subgraph: OF_DEF + AT_STEP + TRIGGERED_BY all present
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"}) OPTIONAL MATCH (r)-[:OF_DEF]->(d:WorkflowDefSnapshot) OPTIONAL MATCH (r)-[:AT_STEP]->(a:Step) OPTIONAL MATCH (r)-[:TRIGGERED_BY]->(m:Message) RETURN "def="+d.key+" at="+a.key+" trig="+m.msgId AS s')
assert_contains "§12.1 start subgraph: OF_DEF + AT_STEP + TRIGGERED_BY" "def=wf_review at=start trig=m1" "$out"

# start_run anchors on indexes (snapshot.key + message.msgId), no label scan
prof=$(gp "$WS" 'MATCH (snap:WorkflowDefSnapshot {key:"wf_review", version:"1"})-[:START]->(start:Step) MATCH (trigger:Message {msgId:"m1"}) RETURN start.key, trigger.msgId')
assert_index_scan "§12.1 start_run anchoring MATCHes use indexes" "$prof"

# runId UNIQUE constraint backstop
out=$(gq "$WS" 'CREATE (:WorkflowRun {runId:"wr1"})' 2>&1)
assert_contains "§12.1 WorkflowRun.runId constraint blocks duplicate" "unique constraint violation" "$out"

echo ""
echo "▶ §12.2 record_step_and_advance (M4 tail-anchored atomic advance)"

ADV='MATCH (r:WorkflowRun {runId:$runId})-[atRel:AT_STEP]->(cur:Step) MATCH (to:Step {stepUid:$toStepUid}) OPTIONAL MATCH (r)-[lastRel:LAST_STEP_RUN]->(prevSR:StepRun) CREATE (sr:StepRun {stepRunId:$stepRunId, stepKey:cur.key, status:$stepStatus, startedAt:$startedAt, endedAt:$endedAt, input:$input, output:$output}) CREATE (r)-[:HAS_STEP_RUN]->(sr) CREATE (sr)-[:RAN]->(cur) FOREACH (p IN CASE WHEN prevSR IS NULL THEN [] ELSE [prevSR] END | CREATE (p)-[:NEXT]->(sr)) FOREACH (lr IN CASE WHEN lastRel IS NULL THEN [] ELSE [lastRel] END | DELETE lr) CREATE (r)-[:LAST_STEP_RUN]->(sr) DELETE atRel CREATE (r)-[:AT_STEP]->(to) SET r.stepCount = r.stepCount + 1 RETURN "sc="+toString(r.stepCount)+" ran="+cur.key+" sr="+sr.stepRunId AS s'

# advance 1: start ran -> move to gather (first advance: seeds the tail, no NEXT)
out=$(gq "$WS" "CYPHER runId=\"wr1\" stepRunId=\"wsr1\" stepStatus=\"done\" startedAt=7001 endedAt=7002 input=\"{}\" output=\"start out\" toStepUid=\"wf_review:1:gather\" $ADV")
assert_contains "§12.2 advance 1: start ran, stepCount 1" "sc=1 ran=start sr=wsr1" "$out"

# state after advance 1: AT_STEP moved to gather (exactly one), tail=wsr1, zero NEXT
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[:AT_STEP]->(a:Step) WITH r, count(a) AS atc, collect(a.key)[0] AS atk MATCH (r)-[:LAST_STEP_RUN]->(l:StepRun) RETURN "atc="+toString(atc)+" at="+atk+" tail="+l.stepRunId AS s')
assert_contains "§12.2 after adv1: exactly one AT_STEP=gather, tail=wsr1" "atc=1 at=gather tail=wsr1" "$out"
out=$(gq "$WS" 'MATCH (:StepRun)-[n:NEXT]->(:StepRun) RETURN count(n) AS n')
assert_contains "§12.2 after adv1: zero NEXT edges (first step-run)" "0" "$out"

# advance 2: gather ran -> move to decide (tail relink: NEXT wsr1->wsr2, tail moves)
out=$(gq "$WS" "CYPHER runId=\"wr1\" stepRunId=\"wsr2\" stepStatus=\"done\" startedAt=7003 endedAt=7004 input=\"{}\" output=\"gather out\" toStepUid=\"wf_review:1:decide\" $ADV")
assert_contains "§12.2 advance 2: gather ran, stepCount 2" "sc=2 ran=gather sr=wsr2" "$out"

# state after advance 2: AT_STEP=decide, NEXT wsr1->wsr2, exactly one tail=wsr2, 2 HAS_STEP_RUN
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[:AT_STEP]->(a:Step) RETURN a.key AS at')
assert_contains "§12.2 after adv2: AT_STEP moved to decide" "decide" "$out"
out=$(gq "$WS" 'MATCH (a:StepRun)-[:NEXT]->(b:StepRun) RETURN "next="+a.stepRunId+"->"+b.stepRunId AS s')
assert_contains "§12.2 after adv2: NEXT appended wsr1->wsr2" "next=wsr1->wsr2" "$out"
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[l:LAST_STEP_RUN]->(sr) RETURN "tailc="+toString(count(l))+" tail="+collect(sr.stepRunId)[0] AS s')
assert_contains "§12.2 after adv2: exactly one tail, moved to wsr2" "tailc=1 tail=wsr2" "$out"
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[:HAS_STEP_RUN]->(sr) RETURN count(sr) AS n')
assert_contains "§12.2 after adv2: 2 HAS_STEP_RUN members" "2" "$out"
out=$(gq "$WS" 'MATCH (sr:StepRun)-[:RAN]->(s:Step) RETURN collect(sr.stepRunId+":"+s.key) AS ran ORDER BY ran')
assert_contains "§12.2 RAN edges: wsr1 ran start" "wsr1:start" "$out"
assert_contains "§12.2 RAN edges: wsr2 ran gather" "wsr2:gather" "$out"

# advance is edge-anchored: index scan on WorkflowRun + Step, no label scan (profiled on a throwaway run)
gq "$WS" 'MATCH (snap:WorkflowDefSnapshot {key:"wf_review", version:"1"})-[:START]->(start:Step) MATCH (trigger:Message {msgId:"m1"}) CREATE (r:WorkflowRun {runId:"wr_prof", defKey:"wf_review", defVersion:"1", status:"running", startedAt:7000, ctx:"{}", trace:false, maxSteps:12, stepCount:0, waitingThreadId:""}) CREATE (r)-[:OF_DEF]->(snap) CREATE (r)-[:AT_STEP]->(start) CREATE (r)-[:TRIGGERED_BY]->(trigger)' > /dev/null
prof=$(gp "$WS" "CYPHER runId=\"wr_prof\" stepRunId=\"wpsr1\" stepStatus=\"done\" startedAt=7001 endedAt=7002 input=\"{}\" output=\"o\" toStepUid=\"wf_review:1:gather\" $ADV")
assert_index_scan "§12.2 record_step_and_advance is edge/index-anchored" "$prof"
gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr_prof"}) OPTIONAL MATCH (r)-[:HAS_STEP_RUN]->(sr) DETACH DELETE r, sr' > /dev/null

echo ""
echo "▶ §12.3/§12.4 suspend / resume — guarded single-flight CAS"

# 12.3 suspend_run: running -> waiting
out=$(gq "$WS" 'CYPHER runId="wr1" threadId="th1" MATCH (r:WorkflowRun {runId:$runId}) WHERE r.status = "running" SET r.status="waiting", r.waitingThreadId=$threadId RETURN "status="+r.status AS s')
assert_contains "§12.3 suspend_run CAS running->waiting" "status=waiting" "$out"

# suspend again (already waiting) -> zero rows (CAS no-op)
out=$(gq "$WS" 'CYPHER runId="wr1" threadId="th1" MATCH (r:WorkflowRun {runId:$runId}) WHERE r.status = "running" SET r.status="waiting", r.waitingThreadId=$threadId RETURN "status="+r.status AS s')
assert_not_contains "§12.3 second suspend is a no-op (zero rows)" "status=waiting" "$out"

# 12.9 find_waiting_run_for_thread: th1 -> wr1, index-anchored
out=$(gq "$WS" 'CYPHER threadId="th1" MATCH (r:WorkflowRun {status:"waiting"}) WHERE r.waitingThreadId = $threadId RETURN r.runId AS runId LIMIT 1')
assert_contains "§12.9 find_waiting_run_for_thread finds wr1" "wr1" "$out"
prof=$(gp "$WS" 'CYPHER threadId="th1" MATCH (r:WorkflowRun {status:"waiting"}) WHERE r.waitingThreadId = $threadId RETURN r.runId LIMIT 1')
assert_index_scan "§12.9 resume lookup anchors on WorkflowRun.status index (no label scan)" "$prof"

# 12.4 resume_run: waiting -> running (single-flight)
out=$(gq "$WS" 'CYPHER runId="wr1" MATCH (r:WorkflowRun {runId:$runId}) WHERE r.status = "waiting" SET r.status="running", r.waitingThreadId="" RETURN "status="+r.status AS s')
assert_contains "§12.4 resume_run CAS waiting->running (winner)" "status=running" "$out"

# concurrent second resume -> zero rows (single-flight guard)
out=$(gq "$WS" 'CYPHER runId="wr1" MATCH (r:WorkflowRun {runId:$runId}) WHERE r.status = "waiting" SET r.status="running", r.waitingThreadId="" RETURN "status="+r.status AS s')
assert_not_contains "§12.4 second concurrent resume is a no-op (single-flight)" "status=running" "$out"

echo ""
echo "▶ §12.6 link_step_emission — StepRun -[:PRODUCED]-> Message (D2, distinct from EMITTED)"

# wsr2 produced the ag2 message (any existing message in th1)
out=$(gq "$WS" 'CYPHER stepRunId="wsr2" msgId="ag2" MATCH (sr:StepRun {stepRunId:$stepRunId}) MATCH (m:Message {msgId:$msgId}) MERGE (sr)-[:PRODUCED]->(m) RETURN sr.stepRunId+"->"+m.msgId AS s')
assert_contains "§12.6 link_step_emission wsr2 PRODUCED ag2" "wsr2->ag2" "$out"

# idempotent MERGE: re-link creates no duplicate
gq "$WS" 'CYPHER stepRunId="wsr2" msgId="ag2" MATCH (sr:StepRun {stepRunId:$stepRunId}) MATCH (m:Message {msgId:$msgId}) MERGE (sr)-[:PRODUCED]->(m) RETURN 1' > /dev/null
out=$(gq "$WS" 'MATCH (:StepRun {stepRunId:"wsr2"})-[e:PRODUCED]->(:Message {msgId:"ag2"}) RETURN count(e) AS n')
assert_contains "§12.6 PRODUCED link is idempotent (exactly one edge)" "1" "$out"

# PRODUCED is distinct from EMITTED: no StepRun has an EMITTED edge (EMITTED is Message->Message, §10)
out=$(gq "$WS" 'MATCH (sr:StepRun)-[e:EMITTED]->() RETURN count(e) AS n')
assert_contains "§12.6 PRODUCED distinct from EMITTED (no StepRun EMITTED edge)" "0" "$out"

echo ""
echo "▶ §12.7/§12.8 get_run + read_step_runs (NEXT-ordered audit trail)"

# 12.7 get_run
out=$(gq "$WS" 'CYPHER runId="wr1" MATCH (r:WorkflowRun {runId:$runId}) OPTIONAL MATCH (r)-[:AT_STEP]->(cur:Step) OPTIONAL MATCH (r)-[:OF_DEF]->(snap:WorkflowDefSnapshot) RETURN "status="+r.status+" sc="+toString(r.stepCount)+" at="+cur.key+" def="+snap.key AS s')
assert_contains "§12.7 get_run returns status, stepCount, atStep, def" "status=running sc=2 at=decide def=wf_review" "$out"

# 12.8 read_step_runs: NEXT-ordered (head-find via OPTIONAL MATCH + IS NULL, then walk)
RSR='CYPHER runId="wr1" MATCH (r:WorkflowRun {runId:$runId})-[:HAS_STEP_RUN]->(sr:StepRun) OPTIONAL MATCH (pv:StepRun)-[:NEXT]->(sr) WITH sr, pv WHERE pv IS NULL MATCH (sr)-[:NEXT*0..]->(x:StepRun) RETURN x.stepRunId AS stepRunId, x.stepKey AS stepKey ORDER BY x.startedAt'
out=$(gq "$WS" "$RSR")
assert_contains "§12.8 read_step_runs returns wsr1 (start)" "wsr1" "$out"
assert_contains "§12.8 read_step_runs returns wsr2 (gather)" "wsr2" "$out"
wsr1_line=$(echo "$out" | grep -n "wsr1" | head -1 | cut -d: -f1)
wsr2_line=$(echo "$out" | grep -n "wsr2" | head -1 | cut -d: -f1)
if [ -n "$wsr1_line" ] && [ -n "$wsr2_line" ] && [ "$wsr1_line" -lt "$wsr2_line" ]; then
  echo "  ✓ §12.8 read_step_runs is NEXT-ordered (wsr1 before wsr2)"
  PASS=$((PASS+1))
else
  echo "  ✗ §12.8 read_step_runs order wrong (wsr1 ${wsr1_line}, wsr2 ${wsr2_line})"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi
prof=$(gp "$WS" "$RSR")
assert_index_scan "§12.8 read_step_runs anchors on WorkflowRun.runId index" "$prof"

echo ""
echo "▶ §12.10/§12.11 trace write/read (debug) + non-debug is empty"

# 12.10 append_trace_event x2 on wsr1
gq "$WS" 'CYPHER stepRunId="wsr1" traceId="wtr1" seq=0 kind="guard_judgment" at=7001 payload="verdict=false; needs more info" MATCH (sr:StepRun {stepRunId:$stepRunId}) CREATE (te:TraceEvent {traceId:$traceId, seq:$seq, kind:$kind, at:$at, payload:$payload}) CREATE (sr)-[:TRACED]->(te) RETURN te.traceId' > /dev/null
out=$(gq "$WS" 'CYPHER stepRunId="wsr1" traceId="wtr2" seq=1 kind="node_rationale" at=7002 payload="asked clarifying question" MATCH (sr:StepRun {stepRunId:$stepRunId}) CREATE (te:TraceEvent {traceId:$traceId, seq:$seq, kind:$kind, at:$at, payload:$payload}) CREATE (sr)-[:TRACED]->(te) RETURN te.traceId AS t')
assert_contains "§12.10 append_trace_event writes a TraceEvent" "wtr2" "$out"

# TraceEvent.traceId UNIQUE constraint blocks a duplicate
out=$(gq "$WS" 'CREATE (:TraceEvent {traceId:"wtr1"})' 2>&1)
assert_contains "§12.10 TraceEvent.traceId constraint blocks duplicate" "unique constraint violation" "$out"

# 12.11 read_trace: ordered by (StepRun.startedAt, TraceEvent.seq)
RT='CYPHER runId="wr1" MATCH (r:WorkflowRun {runId:$runId})-[:HAS_STEP_RUN]->(sr:StepRun)-[:TRACED]->(te:TraceEvent) RETURN te.kind AS kind, te.payload AS payload ORDER BY sr.startedAt, te.seq'
out=$(gq "$WS" "$RT")
assert_contains "§12.11 read_trace returns guard_judgment event" "guard_judgment" "$out"
assert_contains "§12.11 read_trace returns node_rationale event" "node_rationale" "$out"
gj_line=$(echo "$out" | grep -n "guard_judgment" | head -1 | cut -d: -f1)
nr_line=$(echo "$out" | grep -n "node_rationale" | head -1 | cut -d: -f1)
if [ -n "$gj_line" ] && [ -n "$nr_line" ] && [ "$gj_line" -lt "$nr_line" ]; then
  echo "  ✓ §12.11 read_trace ordered by seq (guard_judgment seq0 before node_rationale seq1)"
  PASS=$((PASS+1))
else
  echo "  ✗ §12.11 read_trace order wrong"
  echo "    got: ${out}"
  FAIL=$((FAIL+1))
fi
prof=$(gp "$WS" "$RT")
assert_index_scan "§12.11 read_trace anchors on WorkflowRun.runId index" "$prof"

# non-debug run records zero trace: a fresh run with no append writes no TraceEvent (AC-5 negative)
gq "$WS" 'MATCH (snap:WorkflowDefSnapshot {key:"wf_review", version:"1"})-[:START]->(start:Step) MATCH (trigger:Message {msgId:"m1"}) CREATE (r:WorkflowRun {runId:"wr_lean", defKey:"wf_review", defVersion:"1", status:"running", startedAt:7000, ctx:"{}", trace:false, maxSteps:12, stepCount:0, waitingThreadId:""}) CREATE (r)-[:OF_DEF]->(snap) CREATE (r)-[:AT_STEP]->(start) CREATE (r)-[:TRIGGERED_BY]->(trigger)' > /dev/null
out=$(gq "$WS" 'CYPHER runId="wr_lean" MATCH (r:WorkflowRun {runId:$runId})-[:HAS_STEP_RUN]->(:StepRun)-[:TRACED]->(te:TraceEvent) RETURN count(te) AS n')
assert_contains "§12.11 non-debug run read_trace is empty (AC-5 negative)" "0" "$out"
gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr_lean"}) DETACH DELETE r' > /dev/null

echo ""
echo "▶ §12.5 fail_run (step budget) + complete_run — terminal clears AT_STEP, retains trail"

# step-budget fail: seed wr3 with stepCount 13 > maxSteps 12, fail_run
gq "$WS" 'MATCH (snap:WorkflowDefSnapshot {key:"wf_review", version:"1"})-[:START]->(start:Step) MATCH (trigger:Message {msgId:"m1"}) CREATE (r:WorkflowRun {runId:"wr3", defKey:"wf_review", defVersion:"1", status:"running", startedAt:7000, ctx:"{}", trace:false, maxSteps:12, stepCount:13, waitingThreadId:""}) CREATE (r)-[:OF_DEF]->(snap) CREATE (r)-[:AT_STEP]->(start) CREATE (r)-[:TRIGGERED_BY]->(trigger)' > /dev/null
out=$(gq "$WS" 'CYPHER runId="wr3" endedAt=7100 ctx="{\"error\":\"step budget exceeded\"}" MATCH (r:WorkflowRun {runId:$runId}) OPTIONAL MATCH (r)-[atRel:AT_STEP]->() DELETE atRel SET r.status="failed", r.endedAt=$endedAt, r.ctx=$ctx RETURN "status="+r.status AS s')
assert_contains "§12.5 fail_run (step budget) sets failed" "status=failed" "$out"
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr3"})-[:AT_STEP]->() RETURN count(*) AS n')
assert_contains "§12.5 fail_run clears AT_STEP" "0" "$out"

# complete_run: wr1 (at decide) -> done, AT_STEP cleared, StepRun trail retained
out=$(gq "$WS" 'CYPHER runId="wr1" endedAt=7200 MATCH (r:WorkflowRun {runId:$runId}) OPTIONAL MATCH (r)-[atRel:AT_STEP]->() DELETE atRel SET r.status="done", r.endedAt=$endedAt RETURN "status="+r.status AS s')
assert_contains "§12.5 complete_run sets done" "status=done" "$out"
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[:AT_STEP]->() RETURN count(*) AS n')
assert_contains "§12.5 complete_run clears AT_STEP" "0" "$out"
out=$(gq "$WS" 'MATCH (r:WorkflowRun {runId:"wr1"})-[:HAS_STEP_RUN]->(sr) RETURN count(sr) AS n')
assert_contains "§12.5 complete_run retains the StepRun audit trail (2)" "2" "$out"

# re-complete (AT_STEP already null) — DELETE of a null OPTIONAL edge must not error
out=$(gq "$WS" 'CYPHER runId="wr1" endedAt=7201 MATCH (r:WorkflowRun {runId:$runId}) OPTIONAL MATCH (r)-[atRel:AT_STEP]->() DELETE atRel SET r.status="done", r.endedAt=$endedAt RETURN "status="+r.status AS s' 2>&1)
assert_contains "§12.5 re-complete with null AT_STEP is a no-op (no error)" "status=done" "$out"

# terminal-move on a missing run -> zero rows (WorkflowRunNotFound)
out=$(gq "$WS" 'CYPHER runId="ghost" endedAt=1 MATCH (r:WorkflowRun {runId:$runId}) OPTIONAL MATCH (r)-[atRel:AT_STEP]->() DELETE atRel SET r.status="done" RETURN "status="+r.status AS s')
assert_not_contains "§12.5 complete of a missing run is zero rows" "status=done" "$out"

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
