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
    # --- pass 1 mutants, strings updated to the m2 reader ---
    ("M1 drop -e FALKORDB_ARGS (detached)", S, DET, DET.replace('    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n', "")),
    ("M2 drop -e FALKORDB_ARGS (foreground)", S, FG, FG.replace('    -e FALKORDB_ARGS="${FALKORDB_ARGS}" \\\n', "")),
    ("M3 env file wins over caller env", S, 'if [ -z "${!key+x}" ]; then printf', 'if true; then printf'),
    ("M4 full FALKORDB_ARGS ignored", S, 'if [ -z "${FALKORDB_ARGS:-}" ]; then', 'if true; then'),
    ("M5 no missing-file check", S, 'if [ ! -f "$FALKORDB_ENV_FILE" ] || [ ! -r "$FALKORDB_ENV_FILE" ]; then', 'if false; then'),
    ("M6 unknown key accepted", S, "*) echo \"start_falkordb.sh: unknown key", "*) ;; esac; case x in *) echo \"start_falkordb.sh: unknown key"),
    ("M7 last line w/o newline dropped", S, 'read -r line || [ -n "$line" ]', 'read -r line'),
    ("M8 no CR strip", S, "    line=\"${line%$'\\r'}\"\n", ""),
    ("M9 volume hard-coded", S, '-v "${FALKORDB_VOLUME}":', '-v falkordb-data:'),
    ("M10 MODULE_ARGS may be empty", S, '${FALKORDB_MODULE_ARGS:?set in $FALKORDB_ENV_FILE}', '${FALKORDB_MODULE_ARGS:-}'),
    ("M11 value eval'd (sourced semantics)", S, "printf -v \"$key\" '%s' \"$value\"", 'eval "$key=\\"$value\\""'),
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
    # --- m2 / n1 / n2 refusals (pass 2) ---
    ("M21 duplicate-key check removed", S, 'case "$seen" in *" $key "*)', 'case "$seen" in *" NEVER "*)'),
    ("M22 seen-set never updated", S, '    seen="$seen$key "\n', ""),
    ("M23 value class check removed", S, '|| [[ ! "$value" =~ $value_re ]]', ''),
    ("M24 no-'=' check removed", S, 'if [ "$line" = "$key" ] || ', 'if '),
    ("M25 reviewer's class ^[A-Za-z0-9_. -]*$", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_. -]*$'"),
    ("M26 class admits quotes", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_.\"-]+( [A-Za-z0-9_.\"-]+)*$'"),
    ("M27 class admits #", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_.#-]+( [A-Za-z0-9_.#-]+)*$'"),
    ("M28 class admits $", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_.$-]+( [A-Za-z0-9_.$-]+)*$'"),
    ("M29 class admits =", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_.=-]+( [A-Za-z0-9_.=-]+)*$'"),
    ("M30 class unanchored", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='[A-Za-z0-9_.-]+'"),
    ("M31 class too strict (no dot/dash)", S, "value_re='^[A-Za-z0-9_.-]+( [A-Za-z0-9_.-]+)*$'", "value_re='^[A-Za-z0-9_]+( [A-Za-z0-9_]+)*$'"),
    ("M32 split on last '=' (key%=*)", S, 'key="${line%%=*}"', 'key="${line%=*}"'),
    ("M33 IFS='=' read (old reader, n1)", S, "IFS= read -r line", "IFS='=' read -r line"),
    ("M34 n2: dir accepted as env file", S, 'if [ ! -f "$FALKORDB_ENV_FILE" ] || [ ! -r', 'if [ ! -r'),
    ("M35 n2: old 'not found' message", S, "not found or not readable", "not found"),
    ("M36 default IFS (edge whitespace trimmed)", S, "while IFS= read -r line", "while read -r line"),
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
