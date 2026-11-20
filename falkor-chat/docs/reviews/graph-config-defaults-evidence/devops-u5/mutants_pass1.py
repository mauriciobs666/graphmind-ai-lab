"""U5 mutation run: each mutant is applied to a COPY of the falkor-chat files the test reads
(scripts/start_falkordb.sh, config/falkordb.env, compose.yaml + the test file) in a scratch tree;
the repo's own files are never modified. Usage: python3 mutants.py <falkor-chat dir> <venv python>"""
import shutil, subprocess, sys, tempfile
from pathlib import Path

FC, PY = Path(sys.argv[1]), sys.argv[2]
S, E, C = "scripts/start_falkordb.sh", "config/falkordb.env", "compose.yaml"
DET = '    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n    -v "${FALKORDB_VOLUME}":/var/lib/falkordb/data \\\n    --rm -d'
FG = '    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n    -v "${FALKORDB_VOLUME}":/var/lib/falkordb/data \\\n    --rm -it'
MUTANTS = [
    ("M1 drop -e FALKORDB_ARGS (detached)", S, DET, DET.replace('    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n', "")),
    ("M2 drop -e FALKORDB_ARGS (foreground)", S, FG, FG.replace('    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n', "")),
    ("M3 env file wins over caller env", S, 'if [ -z "${!key+x}" ]; then printf', 'if true; then printf'),
    ("M4 full FALKORDB_ARGS ignored", S, 'if [ -z "${FALKORDB_ARGS:-}" ]; then', 'if true; then'),
    ("M5 no missing-file check", S, 'if [ ! -r "$FALKORDB_ENV_FILE" ]; then', 'if false; then'),
    ("M6 unknown key accepted", S, "*) echo \"start_falkordb.sh: unknown key", "*) continue; echo \"start_falkordb.sh: unknown key"),
    ("M7 last line w/o newline dropped", S, 'read -r key value || [ -n "$key" ]', 'read -r key value'),
    ("M8 no CR strip", S, "\"${value%$'\\r'}\"", '"$value"'),
    ("M9 volume hard-coded", S, '-v "${FALKORDB_VOLUME}":', '-v falkordb-data:'),
    ("M10 MODULE_ARGS may be empty", S, '${FALKORDB_MODULE_ARGS:?set in $FALKORDB_ENV_FILE}', '${FALKORDB_MODULE_ARGS:-}'),
    ("M11 value eval'd (sourced semantics)", S, "printf -v \"$key\" '%s' \"${value%$'\\r'}\"", 'eval "$key=\\"$value\\""'),
    ("M12 banner drops module args", S, 'echo "Module args:             ${FALKORDB_ARGS}"', ':'),
    ("M13 env ASYNC_DELETE 1", E, "ASYNC_DELETE yes", "ASYNC_DELETE 1"),
    ("M14 env MAX_QUEUED_QUERIES 25", E, "MAX_QUEUED_QUERIES 320", "MAX_QUEUED_QUERIES 25"),
    ("M15 env QUERY_MEM_CAPACITY 0", E, "QUERY_MEM_CAPACITY 67108864", "QUERY_MEM_CAPACITY 0"),
    ("M16 env OMP_THREAD_COUNT 8", E, "OMP_THREAD_COUNT 1", "OMP_THREAD_COUNT 8"),
    ("M17 env drops TIMEOUT/RESULTSET (image keys)", E, " TIMEOUT 1000 RESULTSET_SIZE 10000", ""),
    ("M18 compose drops THREAD_COUNT token", C, 'FALKORDB_ARGS: "THREAD_COUNT ${', 'FALKORDB_ARGS: "${'),
    ("M19 compose defaults both vars (no refusal)", C,
     '${FALKORDB_THREAD_COUNT:?run docker compose --env-file config/falkordb.env} ${FALKORDB_MODULE_ARGS:?run docker compose --env-file config/falkordb.env}',
     '${FALKORDB_THREAD_COUNT:-8} ${FALKORDB_MODULE_ARGS:-}'),
    ("M20 compose defaults THREAD_COUNT only", C, '${FALKORDB_THREAD_COUNT:?run docker compose --env-file config/falkordb.env}', '${FALKORDB_THREAD_COUNT:-8}'),
]

def tree(tmp):
    root = tmp / "falkor-chat"
    for rel in (S, E, C, "server/tests/test_start_falkordb_script.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FC / rel, root / rel)
    return root

def pytest(root):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(root / "server"),
                        str(root / "server/tests/test_start_falkordb_script.py")],
                       capture_output=True, text=True, cwd=root / "server")
    return r.returncode, r.stdout.strip().splitlines()[-1]

with tempfile.TemporaryDirectory() as t:
    print("baseline:", pytest(tree(Path(t) / "base")))
killed = 0
for name, rel, old, new in MUTANTS:
    with tempfile.TemporaryDirectory() as t:
        root = tree(Path(t))
        f = root / rel; s = f.read_text()
        assert s.count(old) >= 1, (name, s.count(old))  # all occurrences mutated (M9: both branches)
        f.write_text(s.replace(old, new))
        rc, last = pytest(root)
        killed += rc != 0
        print(f"{'KILLED ' if rc else 'SURVIVED'} {name}: {last}")
print(f"{killed}/{len(MUTANTS)} killed")
