import re, sys
sys.path.insert(0, "/home/mauricio/prg/graphmind-ai-lab/falkor-chat/server")
from falkorchat.repository import Repository, MessageWriteStatus
cap = {}
class G:
    def __init__(s, k): s.k = k
    def query(s, q, p=None):
        cap[s.k] = q
        class R: result_set = [[True, False, False, True, True]]
        return R()
r = Repository.__new__(Repository)
for name in ["post_first_message", "post_subsequent_message"]:
    r._graph = lambda ws, n=name: G(n)
    getattr(r, name)("x", thread_id="t", msg_id="m", author_id="a", text="x", role="user", created_at=1)
def norm(q):
    q = q.replace("\\$", "$")
    q = re.sub(r"^CYPHER(\s+\w+=(\[[^\]]*\]|'[^']*'|\S+))+\s+", "", q.strip())
    q = q.split("RETURN")[0]
    q = re.sub(r"'[^']*'", "L", q); q = re.sub(r"\$\w+", "L", q); q = re.sub(r"\b\d+\b", "L", q)
    return re.sub(r"\s+", "", q)
canon = {"first": norm(cap["post_first_message"]), "subseq": norm(cap["post_subsequent_message"])}
src = open("/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/k094/impl/old_tq.sh").read().splitlines()
for i, line in enumerate(src, 1):
    if "FOREACH (mem IN mems" not in line or "seedIds" in line: continue
    m = re.search(r'g[qp] "\$WS" "(.*)"\)\s*$', line)
    if not m: print(i, "UNPARSED"); continue
    body = m.group(1)
    n = norm(body)
    kind = "first" if "[:HEAD]->(h)" in body else "subseq"
    pre = re.match(r"CYPHER((?:\s+\w+=(?:\[[^\]]*\]|'[^']*'|\S+))+)", body)
    ret = body.split("RETURN",1)[1][:160]
    print(i, kind, "MATCH" if n == canon[kind] else "DIFF", "|", pre.group(1).strip() if pre else None, "|", ret.replace("\\$","$")[:150])
    if n != canon[kind]:
        a, b = n, canon[kind]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print("   suite:", a[max(0,k-60):k+80]); print("   canon:", b[max(0,k-60):k+80])
