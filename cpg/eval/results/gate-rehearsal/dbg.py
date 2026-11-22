"""Debug wrapper: runs gate_run.main with the real default deps; on an exception prints its class and the repo-local frames (file:line:function), never its message."""
import sys, traceback, os
sys.path.insert(0, os.path.join(os.environ["REPO"], "cpg/eval/tools"))
import gate_run
orig = gate_run._run_gate
def wrapped(*a, **k):
    try:
        return orig(*a, **k)
    except BaseException as e:
        fr = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_tb(e.__traceback__) if "site-packages" not in f.filename]
        print("DBG", type(e).__name__, fr[-8:], file=sys.stderr)
        raise
gate_run._run_gate = wrapped
sys.exit(gate_run.main(sys.argv[1:]))
