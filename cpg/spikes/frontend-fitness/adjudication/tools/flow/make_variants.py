"""Regenerate the probe scripts rdrelax.sc (R1), rdwide.sc (upper bound) and rdfix.sc (R2) from
cpg/spikes/frontend-fitness/s28-extract-bodiless-guard.sc. Usage: python3 make_variants.py [outdir] (default: this dir)."""
import os, runpy, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
GUARD = HERE.parents[2] / "s28-extract-bodiless-guard.sc"
out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
out.mkdir(parents=True, exist_ok=True)
shutil.copy(GUARD, out / "rdrelax.sc")
os.chdir(out)
for step in ("step1_relax.py", "step2_wide.py", "step3_fix.py"):   # each edits the scripts in the cwd
    runpy.run_path(str(HERE / step))
print("wrote", out / "rdrelax.sc", out / "rdwide.sc", out / "rdfix.sc")
