import os
"""Compare every ```cypher block in QUERIES.md §12 (at a git rev) with repository.py's query
strings (at a git rev), whitespace-normalised. Usage: cypher_match.py <docs_rev> <code_rev>"""
import ast, re, subprocess, sys
docs_rev, code_rev = sys.argv[1], sys.argv[2]
show = lambda rev, p: subprocess.run(["git", "show", f"{rev}:{p}"], capture_output=True,
                                     text=True, check=True,
                                     cwd=os.path.expanduser("~/prg/graphmind-ai-lab")).stdout
q = show(docs_rev, "falkor-chat/docs/QUERIES.md")
sec = q[q.index("\n## 12. "): q.index("\n## 13. ")]
norm = lambda s: re.sub(r"\s+", " ", s).strip()
blocks = []
for m in re.finditer(r"```cypher\n(.*?)```", sec, re.S):
    body = "\n".join(re.sub(r"\s//.*$", "", l) for l in m.group(1).splitlines() if not l.strip().startswith("//"))
    head = sec[:m.start()].rfind("\n### ")
    blocks.append((sec[head:head + 80].strip().splitlines()[0], norm(body)))
src = show(code_rev, "falkor-chat/server/falkorchat/repository.py")
tree = ast.parse(src)
queries = {}
for fn in ast.walk(tree):
    if isinstance(fn, ast.FunctionDef):
        for call in ast.walk(fn):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr in ("query", "ro_query") and call.args
                    and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                queries.setdefault(norm(call.args[0].value), []).append(fn.name)
ok = bad = 0
for head, body in blocks:
    hit = queries.get(body)
    if hit:
        ok += 1; print(f"MATCH   {head[:60]:60} -> {hit}")
    else:
        bad += 1; print(f"NOMATCH {head[:60]:60}\n        doc: {body[:200]}")
print(f"blocks={len(blocks)} match={ok} nomatch={bad}")
