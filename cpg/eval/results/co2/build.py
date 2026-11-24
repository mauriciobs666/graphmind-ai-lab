"""CO-2 build driver (copy of co1/build.py): the proving-system build (full refresh of the monorepo at a committed HEAD clone) on a private throwaway FalkorDB.

Every location is passed explicitly (clone, work, keys, audit, credentials dir, admin password dir); nothing defaults to the developer's
cpg/.ccg-work, cpg/.audit or ~/.config/ccg. Env (all required): P1T_REPO (clone), P1T_OUT (private area), CCG_CREDENTIALS_DIR, CCG_CONFIG_DIR
(holds falkordb-admin.pass), P1T_PORT (>= 16500, not 6379/6390/16390/16400). Writes $P1T_OUT/build.json (counts, statuses, stages) only.
Run: PYTHONPATH=$P1T_REPO/cpg <venv python> build.py
"""
import json, os, sys, time, resource
from pathlib import Path
from collections import Counter

REPO, OUT, PORT = Path(os.environ["P1T_REPO"]), Path(os.environ["P1T_OUT"]), int(os.environ["P1T_PORT"])
assert PORT >= 16500 and PORT not in (6379, 6390, 16390, 16400), PORT
SYSTEM = "graphmind-ai-lab"
for k in ("CCG_CREDENTIALS_DIR", "CCG_CONFIG_DIR"):
    assert Path(os.environ[k]).resolve().is_relative_to(OUT.resolve().parent), k      # private area only
work, keys, audit = OUT / "work", OUT / "keys", OUT / "audit"
for d in (work, keys, audit):
    d.mkdir(exist_ok=True, mode=0o700)
os.environ["CCG_AUDIT_DIR"] = str(audit)            # the sandbox runner reads the audit dir from the environment only (D3)

import ccg.refresh as RF
assert Path(RF.__file__).resolve().is_relative_to(REPO.resolve()), RF.__file__      # the committed clone's code, not the developer tree
from ccg import sysname
from ccg.store import acl, loader as L

pw = (Path(os.environ["CCG_CONFIG_DIR"]) / "falkordb-admin.pass").read_text().strip()
admin = L.connect("127.0.0.1", PORT, username=None, password=pw)
acl.create_users(admin, SYSTEM)
cr = acl.read_credentials(SYSTEM)
bdb = L.connect("127.0.0.1", PORT, username=sysname.acl_user("build", SYSTEM), password=cr["build"])
vdb = L.connect("127.0.0.1", PORT, username=sysname.acl_user("verdict", SYSTEM), password=cr["verdict"])
t0 = time.time()
rep = RF.refresh(SYSTEM, manifest=REPO / "cpg/ccg/system.yaml", repo_root=REPO, work_root=work, keys_root=keys, audit_dir=audit,
                 target=RF.Target(L.LiveStore(bdb, SYSTEM), bdb, vdb))
wall = round(time.time() - t0, 1)
res = {"head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), "wall_s": wall,
       "units": len(rep.units), "status": dict(Counter(u.status for u in rep.units)),
       "by_language_status": {f"{k[0]}/{k[1]}": v for k, v in Counter((u.language, u.status) for u in rep.units).items()},
       "not_fresh": [{"unit": u.unit_id[:12], "root": u.root_path, "lang": u.language, "status": u.status, "reason": u.fail_reason} for u in rep.units if u.status != "fresh"],
       "unknown_kind": rep.unknown_kind, "unattested": len(rep.unattested_units), "bind_key_mismatch": len(rep.bind_key_mismatch),
       "snapshot_id": rep.snapshot_id, "dropped": rep.dropped, "link_counts": rep.link_counts,
       "stages": [{"name": s.name, "s": round(s.seconds, 1), "peak_rss_kb": s.peak_rss_kb, "mem_avail_min_kb": s.mem_available_min_kb, "mem_avail_drop_kb": s.mem_available_drop_kb} for s in rep.stages],
       "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
       "ack": rep.lines()[:2]}
(OUT / "build.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: res[k] for k in ("head", "wall_s", "units", "status", "not_fresh", "snapshot_id")}, default=str))
sys.exit(0 if not rep.not_mapped and not rep.stale_kept else 3)
