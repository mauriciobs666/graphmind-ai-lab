"""U5 m2 evidence: each env-file grammar axis through the edited start_falkordb.sh (copy, docker
stubbed) and through `docker compose --env-file <same file> config` (render only).
Usage: python3 -I grammar_axes.py <falkor-chat dir>"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
FC = Path(sys.argv[1])
sys.path.insert(0, str(FC / "server" / "tests"))
src = (FC / "server/tests/test_start_falkordb_script.py").read_text()
ns = {"__file__": str(FC / "server/tests/test_start_falkordb_script.py"), "__name__": "axes"}
exec(compile(src, "t", "exec"), ns)
docker = shutil.which("docker")
for name, text, expect in ns["_GRAMMAR_AXES"]:
    with tempfile.TemporaryDirectory() as t:
        out, calls = ns["_run"](Path(t), "-d", env_text=text)
        script = ("REFUSED: " + out.stderr.strip().splitlines()[-1].split(": ", 1)[-1]) if out.returncode else \
                 [a for a in calls[0] if a.startswith("FALKORDB_ARGS=")][0][14:]
        f = Path(t) / "falkor-chat/config/falkordb.env"
        r = subprocess.run([docker, "compose", "-p", "k098-devops-probe", "-f", str(FC / "compose.yaml"),
                            "--env-file", str(f), "config", "--format", "json"], capture_output=True, text=True,
                           env={"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin"})
        comp = json.loads(r.stdout)["services"]["falkordb"]["environment"]["FALKORDB_ARGS"] if r.returncode == 0 \
               else "REFUSED: " + r.stderr.strip()[:80]
        verdict = "refused" if out.returncode else ("SAME" if script == comp else "DIFFERS")
        print(f"{name:28} expect={expect:6} {verdict:8} script={script!r}\n{'':44} compose={comp!r}")
