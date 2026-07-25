"""Make `server.py` importable.

`cpg/mcp` is a script run by path, not an installable package, so the module
directory is put on `sys.path` here rather than through a package install. The
path is derived from this file's own location — never hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
