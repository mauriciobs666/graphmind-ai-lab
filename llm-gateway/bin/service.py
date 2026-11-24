"""Entry shim for the gateway service (docs/plans/llm-gateway.md §3.1, §3.4, §4 D8).

Run as `/usr/bin/python3 -I /opt/llmgateway/bin/service.py` (the `service-unit` the installer
writes). `-I` drops PYTHONPATH and the script directory, so this puts the `src/` of its own
root-owned code root on `sys.path` (from the path it was started by, deliberately not resolving
symlinks) and runs `service.main`. The same shape as `keyholder.py`.
"""

import os
import sys

if not sys.flags.isolated:  # without -I, PYTHONPATH and the script directory would be honoured
    sys.exit("service.py must be run as: /usr/bin/python3 -I service.py")

sys.dont_write_bytecode = True  # the code root stays exactly what was installed
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from llmgateway.service import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
