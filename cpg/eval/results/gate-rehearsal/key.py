import sys, os
sys.path.insert(0, os.path.join(os.environ["REPO"], "cpg/eval/tools"))
import gate_run
from pathlib import Path
print(gate_run._read_lists([Path(p) for p in sys.argv[1].split(",")], None)[1])
