#!/usr/bin/env python3
"""Thin entry point; resolves symlinks so the skill's one shared copy is used."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wilib.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
