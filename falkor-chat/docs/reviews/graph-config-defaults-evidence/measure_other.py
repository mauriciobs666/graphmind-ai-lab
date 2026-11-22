import os
"""K-098: QUERY_MEM_CAPACITY needs of the heaviest NON-RAG read (read_thread, 10k msgs) and the
heaviest write (create_document at MAX_DOCUMENT_CHARS) — probe container only. Query texts copied
from repository.py. Also: does a capped write leave partial state?"""
import json, sys, time, uuid
from falkordb import FalkorDB
from redis.exceptions import ResponseError
sys.path.insert(0, os.path.expanduser("~/prg/graphmind-ai-lab/falkor-chat/server"))
from falkorchat.chunking import split_into_chunks
db = FalkorDB(host="127.0.0.1", port=16479); r = db.connection; g = db.select_graph("ws:synth")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
if not g.ro_query("MATCH (u:User {userId:'u1'}) RETURN count(u)").result_set[0][0]:
    q("CREATE (:User {userId:'u1', displayName:'U One'})")
    q("CREATE INDEX FOR (n:User) ON (n.userId)"); q("CREATE INDEX FOR (n:Document) ON (n.documentId)")
    for s in range(0, 10000, 1000):
        q("MATCH (u:User {userId:'u1'}) UNWIND range($s,$s+999) AS i MATCH (m:Message {msgId:'tlong-'+toString(i)}) CREATE (m)-[:POSTED_BY]->(u)", {"s": s})
READ_THREAD = ("MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) "
  "MATCH (m)-[:POSTED_BY]->(author) RETURN m.msgId AS msgId, m.text AS text, m.role AS role, m.createdAt AS createdAt, "
  "coalesce(author.userId, author.agentId) AS authorId, author.displayName AS displayName, labels(author) AS authorType, "
  "coalesce(m.toolsUsed, []) AS toolsUsed ORDER BY m.createdAt")
CREATE_DOC = ("OPTIONAL MATCH (u:User  {userId:  $ingestedBy}) OPTIONAL MATCH (a:Agent {agentId: $ingestedBy}) "
  "WITH u, a, coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok "
  "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | CREATE (d:Document {documentId: $documentId, title: $title, text: $text, "
  "sourceFormat: $sourceFormat, sourceKind: CASE WHEN u IS NOT NULL THEN 'document' ELSE 'agent' END, status: 'processing', "
  "pendingJobs: 0, createdAt: $createdAt, currentVersion: true}) CREATE (d)-[:INGESTED_BY]->(ingestor) "
  "FOREACH (ch IN $chunks | CREATE (d)-[:HAS_CHUNK]->(:Chunk {chunkId: ch.chunkId, text: ch.text, seq: ch.seq, "
  "documentId: $documentId, documentCurrent: true}))) RETURN ok AS written, ingestor IS NOT NULL AS ingestorFound")
text = ("The quick brown fox jumps over the lazy dog. " * 12000)[:500000]
chunks = split_into_chunks(text)
def doc_params():
    did = "doc-" + uuid.uuid4().hex
    return {"documentId": did, "title": "big", "text": text, "sourceFormat": "text", "ingestedBy": "u1",
            "createdAt": 1, "chunks": [{"chunkId": f"{did}-{i}", "text": c, "seq": i} for i, c in enumerate(chunks)]}
def cap(b): r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", str(b))
def attempt(write, c, p):
    try:
        (g.query if write else g.ro_query)(c, p, timeout=60000); return True, None
    except ResponseError as e: return False, str(e)
def bisect(write, c, pf):
    cap(0); lo, hi = 1, 1 << 30
    while hi - lo > 1024:
        m = (lo + hi) // 2; cap(m)
        okk, err = attempt(write, c, pf())
        if okk: hi = m
        else:
            assert "mem consumption" in err, err; lo = m
    cap(0); return hi
out = open(sys.argv[1], "a")
def rec(d): print(json.dumps(d), flush=True); out.write(json.dumps(d) + "\n")
t = time.perf_counter(); n = len(g.ro_query(READ_THREAD, {"threadId": "tlong"}, timeout=60000).result_set); ms = (time.perf_counter() - t) * 1000
rec({"variant": "read_thread tlong (10k msgs, ~200-char text)", "rows": n, "uncapped_ms": round(ms, 1),
     "min_cap_kib": round(bisect(False, READ_THREAD, lambda: {"threadId": "tlong"}) / 1024)})
# partial-state check: a capped write must leave nothing
before = g.ro_query("MATCH (d:Document) RETURN count(d)").result_set[0][0], g.ro_query("MATCH (c:Chunk) RETURN count(c)").result_set[0][0]
cap(1024 * 1024); okk, err = attempt(True, CREATE_DOC, doc_params()); cap(0)
after = g.ro_query("MATCH (d:Document) RETURN count(d)").result_set[0][0], g.ro_query("MATCH (c:Chunk) RETURN count(c)").result_set[0][0]
rec({"variant": "create_document under 1 MiB cap: partial-state check", "ok": okk, "err": err, "docs_chunks_before": before, "after": after})
t = time.perf_counter(); attempt(True, CREATE_DOC, doc_params()); ms = (time.perf_counter() - t) * 1000
rec({"variant": f"create_document MAX_DOCUMENT_CHARS=500000, {len(chunks)} chunks", "uncapped_ms": round(ms, 1),
     "min_cap_kib": round(bisect(True, CREATE_DOC, doc_params) / 1024)})
