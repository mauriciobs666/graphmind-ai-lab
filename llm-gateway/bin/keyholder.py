"""Entry shim for the key holder (docs/plans/llm-gateway.md §4 D7).

Run as `/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py [serve|log|check-deployment|...]`.
`-I` drops PYTHONPATH and the script directory, so this puts the `src/` of its own root-owned
code root on `sys.path` (from the path it was started by, deliberately not resolving symlinks:
`keyholder check-deployment` then sees a symlinked code root as one) and runs `keyholder.main`.
"""

import os
import sys

if not sys.flags.isolated:  # without -I, PYTHONPATH and the script directory would be honoured
    sys.exit("keyholder.py must be run as: /usr/bin/python3 -I keyholder.py")

sys.dont_write_bytecode = (
    True  # the code root stays exactly what was installed (root runs this too)
)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from llmgateway.keyholder import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
