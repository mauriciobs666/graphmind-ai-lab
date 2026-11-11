"""Errors shared by the extractors (C-1001 P1).

``SourceRefused`` is the one class an extractor raises for a source file it will not read; ``normalize.core``
re-exports this same class (identity, not a copy), so an ``except`` on either import path catches both.
"""
from __future__ import annotations


class SourceRefused(RuntimeError):
    """A source file the extractor will not read: its name leaves the stage (``..``, an absolute name, a symlink out), or
    its bytes are not UTF-8, or it cannot be read. The unit fails; nothing is skipped silently.

    Contract: the message names the repo-relative path (capped and escaped by the raiser) and the reason, never file
    contents or a value read from the file."""
