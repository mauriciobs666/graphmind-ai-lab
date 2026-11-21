"""U12 mutation harness: each mutant edits a COPY of falkorchat/ (u12/mut/), never
the real tree; pytest runs the real tests with the copy first on sys.path.
Usage: python mutants.py   (from anywhere; writes mutants.txt next to itself)."""
import os, re, shutil, subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
SERVER = HERE.parents[3] / "server"
REAL = SERVER / "falkorchat"
MUT = HERE / "mut"
PY = SERVER / ".venv/bin/python"
PYTEST = SERVER / ".venv/bin/pytest"  # console script: no cwd on sys.path, so PYTHONPATH wins
TESTS = ["tests/test_graphrag.py", "tests/test_services.py", "tests/test_tools.py",
         "tests/test_responder.py", "tests/test_background.py"]

A = '"MATCH (t:Thread {threadId: seed.threadId}) "'
SUB = ('"CALL { WITH seed "\n'
       '            "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "\n'
       '            "WITH DISTINCT related LIMIT 5 "\n'
       '            "RETURN collect(related) AS expanded } "')
ORD = '"ORDER BY score ASC, msgId ASC "'
SLICE = ('"OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "\n'
         '            "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "')

MUTANTS = {
    "M0 current unbounded form (walk + collect[..5], score-only order)": [
        (A, '"MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "'), (SUB, SLICE), (ORD, '"ORDER BY score ASC "')],
    "M1 rejected alt: anchor+tie kept, collect(DISTINCT)[..5] instead of subquery": [(SUB, SLICE)],
    "M2 LIMIT 5 -> 6": [('"WITH DISTINCT related LIMIT 5 "', '"WITH DISTINCT related LIMIT 6 "')],
    "M3 LIMIT 5 -> 4": [('"WITH DISTINCT related LIMIT 5 "', '"WITH DISTINCT related LIMIT 4 "')],
    "M4 DISTINCT dropped in subquery (equivalent on this build: m4_distinct_probe.txt)": [('"WITH DISTINCT related LIMIT 5 "', '"WITH related LIMIT 5 "')],
    "M5 tie-break dropped": [(ORD, '"ORDER BY score ASC "')],
    "M6 tie-break msgId DESC": [(ORD, '"ORDER BY score ASC, msgId DESC "')],
    "M7 tie-break on text": [(ORD, '"ORDER BY score ASC, text ASC "')],
    "M8 per-seed scoping: LIMIT 5 global, outside any subquery": [(SUB,
        '"OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "\n'
        '            "WITH DISTINCT seed, score, related LIMIT 5 "\n'
        '            "WITH seed, score, collect(related) AS expanded "')],
    "M9 per-seed scoping: subquery does not import seed": [('"CALL { WITH seed "', '"CALL { "')],
    "M10 thread anchor dropped (any Thread)": [(A, '"MATCH (t:Thread) "')],
    "M11 thread anchor on wrong key (msgId)": [(A, '"MATCH (t:Thread {threadId: seed.msgId}) "')],
    "M12 thread anchor reverted to the walk (expansion bounded)": [(A, '"MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "')],
    "M13 thread anchor as MATCH+WHERE (equivalent: identical plan, m13_equivalence_explain.txt)": [(A, '"MATCH (t:Thread) WHERE t.threadId = seed.threadId "')],
    "M14 seed excluded from its own related": [('(related:Message) "', '(related:Message) WHERE related <> seed "')],
    "M15 scoped WITH barrier removed (F1: planner anchors on Channel)": [('"WITH seed, score, t "', '""')],
}

def run(name, edits):
    if MUT.exists(): shutil.rmtree(MUT)
    shutil.copytree(REAL, MUT / "falkorchat", ignore=shutil.ignore_patterns("__pycache__"))
    p = MUT / "falkorchat/repository.py"; s = p.read_text()
    for old, new in edits:
        assert s.count(old) == 1, (name, old)
        s = s.replace(old, new)
    p.write_text(s)
    env = dict(os.environ, PYTHONPATH=str(MUT), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(PYTEST), "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=SERVER, env=env, capture_output=True, text=True)
    # prove the copy, not the real tree, was imported
    chk = subprocess.run([str(PY), "-B", "-c", "import falkorchat.repository as r; print(r.__file__)"],
                         cwd=HERE, env=env, capture_output=True, text=True).stdout.strip()
    failed = sorted(set(re.findall(r"^FAILED (\S+)", r.stdout, re.M)))
    summary = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    return chk, summary, failed

out = []
for name, edits in MUTANTS.items():
    chk, summary, failed = run(name, edits)
    assert chk.startswith(str(MUT)), chk
    out.append(f"## {name}\n  imported: {chk}\n  {summary}\n  " + ("KILLED by:\n    " + "\n    ".join(failed) if failed else "SURVIVED"))
    print(out[-1], flush=True)
shutil.rmtree(MUT)
(HERE / "mutants.txt").write_text("\n".join(out) + "\n")
