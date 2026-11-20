"""K-097 mutation runner. Mutates COPIES under mutroot/server/falkorchat only (pristine
source = the live tree, read-only), runs the targeted tests, restores by copy."""
import shutil, subprocess, sys, pathlib
LIVE = pathlib.Path("/home/mauricio/prg/graphmind-ai-lab/falkor-chat/server/falkorchat")
ROOT = pathlib.Path(sys.argv[1])
PKG = ROOT / "server" / "falkorchat"
PY = "/home/mauricio/prg/graphmind-ai-lab/falkor-chat/server/.venv/bin/python"
TESTS = ["tests/test_k097_review_probe.py", "tests/test_workflow_input_cas.py", "tests/test_process_input.py",
         "tests/test_workflow_timers.py", "tests/test_executor.py",
         "tests/test_repository.py"]
M = {
 "compare dropped": ("repository.py",
   "WHERE r.status = 'waiting' AND coalesce(r.ctxVersion, 0) = $ctxVersion ",
   "WHERE r.status = 'waiting' "),
 "12.13 bump dropped": ("repository.py",
   "r.ctx = $ctx, \"\n            \"    r.ctxVersion = $ctxVersion + 1 ",
   "r.ctx = $ctx \"\n            \" "),
 "resume_run bump dropped": ("repository.py",
   "r.waitingThreadId = '', \"\n            \"    r.ctxVersion = coalesce(r.ctxVersion, 0) + 1 ",
   "r.waitingThreadId = '' \"\n            \" "),
 "resume_run bump = constant 1": ("repository.py",
   "r.waitingThreadId = '', \"\n            \"    r.ctxVersion = coalesce(r.ctxVersion, 0) + 1 ",
   "r.waitingThreadId = '', \"\n            \"    r.ctxVersion = 1 "),
 "fail_run bump dropped": ("repository.py",
   "r.ctx = $ctx, \"\n            \"    r.ctxVersion = coalesce(r.ctxVersion, 0) + 1 ",
   "r.ctx = $ctx \"\n            \" "),
 "compare without coalesce": ("repository.py",
   "AND coalesce(r.ctxVersion, 0) = $ctxVersion", "AND r.ctxVersion = $ctxVersion"),
 "get_run without coalesce": ("repository.py",
   "coalesce(r.ctxVersion, 0) AS ctxVersion", "r.ctxVersion AS ctxVersion"),
 "submit: version re-read at CAS time": ("services.py",
   "ctx_version=run[\"ctxVersion\"],",
   "ctx_version=self._repo.get_run(ctx.ws, run_id=run_id)[\"ctxVersion\"],"),
 "sweep: version re-read at CAS time": ("services.py",
   "v=run[\"ctxVersion\"]: executor.resume(\n                        ctx, run_id=rid, run_ctx_json=mj, ctx_version=v,",
   "v=None: executor.resume(\n                        ctx, run_id=rid, run_ctx_json=mj, ctx_version=self._repo.get_run(ctx.ws, run_id=rid)[\"ctxVersion\"],"),
 "executor guard removed": ("executor.py",
   "if run_ctx_json is not None and ctx_version is None:",
   "if False:"),
}
only = sys.argv[2:] or list(M)
for name in only:
    fname, old, new = M[name]
    src = (LIVE / fname).read_text()
    n = src.count(old)
    assert n == 1, (name, n)
    (PKG / fname).write_text(src.replace(old, new))
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=ROOT / "server", capture_output=True, text=True)
    shutil.copy(LIVE / fname, PKG / fname)  # restore by copy
    failed = sorted({l.split("::")[1].split(" ")[0] for l in r.stdout.splitlines()
                     if l.startswith(("FAILED", "ERROR")) and "::" in l})
    print(f"### {name}: exit={r.returncode} {r.stdout.strip().splitlines()[-1]}")
    for f in failed:
        print("   red:", f)
for f in ("repository.py", "services.py", "executor.py"):
    assert (PKG / f).read_text() == (LIVE / f).read_text(), f
print("copies restored = live tree")
