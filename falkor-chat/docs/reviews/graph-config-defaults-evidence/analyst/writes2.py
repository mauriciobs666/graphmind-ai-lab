import time, uuid
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from falkorchat.repository import Repository
from falkorchat.chunking import split_into_chunks
db=FalkorDB(host="127.0.0.1",port=16593); r=db.connection
print("cap", r.execute_command("GRAPH.CONFIG","GET","QUERY_MEM_CAPACITY"))
repo=Repository(db)
# 1. live ingest write, superseding an identical 500k-char document
g=db.select_graph("ws:an-doc"); g.query("CREATE (:User {userId:'u1'})")
text=("The quick brown fox jumps over the lazy dog. "*12000)[:500000]; chunks=split_into_chunks(text)
for n in (1,2):
    did="doc-"+uuid.uuid4().hex
    t=time.perf_counter()
    try:
        out=repo.create_document_with_auto_supersede("an-doc",document_id=did,title="big",text=text,text_normalized_hash="h",
            source_format="text",ingested_by="u1",created_at=n,chunks=[{"chunkId":f"{did}-{i}","text":c,"seq":i} for i,c in enumerate(chunks)],
            match_id="m"+str(n),lsh_bands=[f"b{i}" for i in range(8)])
        print(f"ingest #{n} ({len(chunks)} chunks) @64MiB ok autoSuperseded={out.get('autoSuperseded')} {time.perf_counter()-t:.2f}s", flush=True)
    except ResponseError as e: print(f"ingest #{n} ERR {e}", flush=True)
# 2. reset_all_participants verbatim
cy=Repository._RESET_ALL_PARTICIPANTS_CYPHER
for P,M in ((1000,10),(5000,10),(10000,10),(20000,10)):
    g=db.select_graph(f"ws:an-r3-{P}")
    for s in range(0,P,1000):
        g.query("UNWIND range($s,$e-1) AS i CREATE (u:User {userId:'p'+toString(i), tokenHash:'h'})-[:MEMBER_OF]->(c:Channel {channelId:'c'+toString(i), participantId:'p'+toString(i)})-[:HAS_THREAD]->(t:Thread {threadId:'t'+toString(i)}) "
                "WITH t, i UNWIND range(0,$m-1) AS j CREATE (m:Message {msgId:'m'+toString(i)+'-'+toString(j), seq:j, text:'hello there this is a storefront message'}) "
                "WITH t, i, m ORDER BY j WITH t, i, collect(m) AS ms WITH t, ms, ms[0] AS h CREATE (t)-[:HEAD]->(h) WITH ms UNWIND range(0,size(ms)-2) AS k WITH ms[k] AS a, ms[k+1] AS b CREATE (a)-[:NEXT]->(b)",{"s":s,"e":min(P,s+1000),"m":M})
    before=g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    t=time.perf_counter()
    try: res=g.query(cy); out="ok msgs="+str(res.result_set[0][3])
    except ResponseError as e: out="ERR "+str(e)
    after=g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    print(f"reset_all @64MiB participants={P} msgs={P*M}: {out} ({time.perf_counter()-t:.2f}s) nodes before={before} after={after}", flush=True)
    if out.startswith("ERR"): break
