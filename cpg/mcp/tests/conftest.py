"""Make `server.py` importable.

`cpg/mcp` is a script run by path, not an installable package, so the module
directory is put on `sys.path` here rather than through a package install. The
path is derived from this file's own location — never hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_MODULE_DIR))

# `test_build_inputs.py` tests the BUILD TOOLING (build.sh / image-tag.sh / Dockerfile),
# which `.dockerignore` deliberately keeps out of the build context — so inside the
# container there is nothing for it to test. Not collecting it there (rather than
# skipping it) keeps the in-image gate's expected counts exactly what they were, so a
# real regression still shows up as a diff instead of hiding among "skipped" lines.
collect_ignore = []
if not (_MODULE_DIR / "build.sh").exists():
    collect_ignore.append("test_build_inputs.py")
