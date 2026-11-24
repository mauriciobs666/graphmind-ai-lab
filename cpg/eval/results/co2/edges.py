import json, os, sys
sys.path[:0]=[os.environ["P1T_REPO"]+"/cpg"]
from clients.common import Client
c=Client("graphmind-ai-lab",host="127.0.0.1",port=int(os.environ["P1T_PORT"]))
q="MATCH (c:CallSite)-[r:CALLS]->(s:Symbol) WHERE s.name='<script>' RETURN coalesce(r.status,'confirmed'), coalesce(r.reason,'resolved-call'), count(r)"
print(c.raw_query(q)["rows"])
q2="MATCH (c:CallSite)-[r:CALLS]->(s:Symbol) WHERE s.name='<script>' MATCH (u:Unit {unitId:c.unitId}) RETURN u.language, count(r)"
print(c.raw_query(q2)["rows"])
