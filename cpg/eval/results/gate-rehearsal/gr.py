"""Rehearsal driver: the REAL gate_run.main with the real default deps, plus env-selected plants that act on the real process.
  W2 (always)     ccg.refresh.refresh is called with the manifest of the copy's HEAD instead of <clone at EVAL_COMMIT>/cpg/ccg/system.yaml, which does not exist at
                  EVAL_COMMIT (defect D2); everything else in the build is the harness's own code.
  KILL_AT=ask:N | runfile | marker-asked   os.kill(self, KILL_SIG or KILL) at the N-th engine ask of this process / before the run.json write / before the final marker write
  FAULT=N:ClassName[,M:ClassName]          the N-th engine ask of this process raises that class (ValueError | TimeoutError | BadRequest | OSError(ENOSPC))
  STALE=N                                  the N-th engine answer is returned with a stale stamp snapshotId (a stale engine)
Nothing under cpg/eval/tools is modified."""
import copy, errno, os, signal, sys, time
sys.path.insert(0, os.path.join(os.environ["REPO"], "cpg/eval/tools"))
import gate_run
from pathlib import Path
import ccg.refresh as R
import clients.common as CC

MANIFEST = Path(os.environ["REPO"]) / "cpg/ccg/system.yaml"
_orig_refresh = R.refresh
R.refresh = lambda *a, **k: _orig_refresh(*a, **{**k, "manifest": MANIFEST})

kill_at = os.environ.get("KILL_AT", "")
sig = {"KILL": signal.SIGKILL, "TERM": signal.SIGTERM}[os.environ.get("KILL_SIG", "KILL")]
faults = {int(p.split(":")[0]): p.split(":")[1] for p in os.environ.get("FAULT", "").split(",") if p}
stale = {int(x) for x in os.environ.get("STALE", "").split(",") if x}
n = [0]
_ask = CC.Client.ask
def ask(self, *a, **k):
    n[0] += 1
    if kill_at == f"ask:{n[0]}":
        os.kill(os.getpid(), sig)
        time.sleep(0.2)
    if n[0] in faults:
        c = faults[n[0]]
        raise {"ValueError": lambda: ValueError("planted"), "TimeoutError": lambda: TimeoutError("planted"),
               "BadRequest": lambda: CC.BadRequest("planted"), "ENOSPC": lambda: OSError(errno.ENOSPC, "planted")}[c]()
    out = _ask(self, *a, **k)
    if n[0] in stale:
        out = copy.deepcopy(out); out["stamp"]["snapshotId"] = "0" * 32
    return out
CC.Client.ask = ask

_aw, _wm = gate_run._atomic_write, gate_run._write_marker
def atomic_write(path, text):
    if kill_at == "runfile" and path.name.startswith("run-") and path.name.endswith(".json"):
        os.kill(os.getpid(), signal.SIGKILL); time.sleep(0.2)
    return _aw(path, text)
def write_marker(path, doc):
    if kill_at == "marker-asked" and doc.get("state") == "asked":
        os.kill(os.getpid(), signal.SIGKILL); time.sleep(0.2)
    return _wm(path, doc)
gate_run._atomic_write, gate_run._write_marker = atomic_write, write_marker
# LINEFAULT=N:ValueError  the N-th answer-line append of this process raises (a fault outside the engine's ask: a run-wide, deterministic one)
_al = gate_run._append_line; lf = [0]
def append_line(path, obj):
    lf[0] += 1
    if os.environ.get("LINEFAULT", "").split(":")[0] == str(lf[0]):
        raise {"ValueError": ValueError}[os.environ["LINEFAULT"].split(":")[1]]("planted")
    return _al(path, obj)
gate_run._append_line = append_line

# timing / memory (item 10): phase wall times, per-ask max/sum, peak RSS of this process; written to $TIMING_FILE (numbers only)
import atexit, json, resource
T = {"ask_n": 0, "ask_sum_s": 0.0, "ask_max_s": 0.0}
def _timed(obj, name, key):
    f = getattr(obj, name)
    def w(*a, **k):
        t = time.monotonic()
        try:
            return f(*a, **k)
        finally:
            T[key] = round(time.monotonic() - t, 2)
    setattr(obj, name, w)
_timed(gate_run.G, "build_at_eval_commit", "build_s")
_timed(gate_run, "resolve_preflight", "resolve_dryrun_s")
_inner = CC.Client.ask
def ask2(self, *a, **k):
    t = time.monotonic()
    try:
        return _inner(self, *a, **k)
    finally:
        d = time.monotonic() - t; T["ask_n"] += 1; T["ask_sum_s"] = round(T["ask_sum_s"] + d, 3); T["ask_max_s"] = round(max(T["ask_max_s"], d), 3)
CC.Client.ask = ask2
@atexit.register
def _dump():
    T["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024); T["peak_rss_children_mb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
    if os.environ.get("TIMING_FILE"):
        open(os.environ["TIMING_FILE"], "w").write(json.dumps(T))
sys.exit(gate_run.main(sys.argv[1:]))
