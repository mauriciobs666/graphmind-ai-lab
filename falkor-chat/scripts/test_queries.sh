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
