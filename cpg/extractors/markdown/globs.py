"""The prompt-bearing Markdown glob list: loaded from ``cpg/eval/eval-manifest.yaml``, never duplicated here.

Census scope equals evaluated scope (eval-protocol §3.3, EP Q3): a ``.md`` file is ``prompt-md`` iff it matches an
``include`` glob and no ``except`` glob of the manifest's ``promptBearingMarkdown`` block; every other Markdown file is
``doc``. The loader fails (``ManifestError``) if the manifest is absent, is not the expected schema, or the block's shape
changed (a wrong key set, an empty ``include``, a pattern outside the accepted glob alphabet), instead of falling back
to a built-in list.

Matching is by hand, not by regex: ``*`` and ``?`` act inside one path segment, a whole ``**`` segment matches zero or
more segments, and both matchers are polynomial in the (capped) pattern and path sizes, so a hostile path cannot cause
backtracking blow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[2] / "eval" / "eval-manifest.yaml"
SCHEMA = "ccg-eval-manifest-v1"
KLASSES = ("prompt-md", "doc")

_MAX_MANIFEST_BYTES = 262_144
_MAX_PATTERNS = 64
_MAX_PATTERN_LEN = 200
_MAX_SEGMENTS = 64
_MAX_SEGMENT_LEN = 255
_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-*?/")


class ManifestError(ValueError):
    """The manifest is absent or its ``promptBearingMarkdown`` block is not the expected shape."""


def _check_patterns(name: str, value: object, *, non_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (non_empty and not value) or len(value) > _MAX_PATTERNS:
        raise ManifestError(f"promptBearingMarkdown.{name} must be a list of 1..{_MAX_PATTERNS} globs"
                            if non_empty else f"promptBearingMarkdown.{name} must be a list of <= {_MAX_PATTERNS} globs")
    out = []
    for p in value:
        if (not isinstance(p, str) or not p or len(p) > _MAX_PATTERN_LEN or not set(p) <= _ALPHABET
                or p.startswith("/") or "//" in p or any(s in ("", ".", "..") for s in p.split("/"))
                or any("**" in s and s != "**" for s in p.split("/"))):
            raise ManifestError(f"promptBearingMarkdown.{name}: unsupported glob {p!r}")
        out.append(p)
    return tuple(out)


@dataclass(frozen=True)
class Globs:
    include: tuple[str, ...]
    exclude: tuple[str, ...]

    def classify(self, path: str) -> str:
        """``prompt-md`` or ``doc`` for a repository-relative POSIX path of a Markdown file."""
        if any(_match(p, path) for p in self.include) and not any(_match(p, path) for p in self.exclude):
            return "prompt-md"
        return "doc"


def _seg_match(pat: str, text: str) -> bool:
    """``*`` (any run) and ``?`` (one char) over one segment; the greedy two-pointer form, O(len(pat) * len(text))."""
    p = t = 0
    star = mark = -1
    while t < len(text):
        if p < len(pat) and pat[p] in ("?", text[t]) and pat[p] != "*":
            p += 1
            t += 1
        elif p < len(pat) and pat[p] == "*":
            star, mark = p, t
            p += 1
        elif star != -1:
            p = star + 1
            mark += 1
            t = mark
        else:
            return False
    while p < len(pat) and pat[p] == "*":
        p += 1
    return p == len(pat)


def _match(pattern: str, path: str) -> bool:
    segs = path.split("/")
    pats = pattern.split("/")
    if len(segs) > _MAX_SEGMENTS or any(len(s) > _MAX_SEGMENT_LEN for s in segs):
        return False
    # memoised over (pattern index, path index): ``**`` consumes zero or more whole segments
    seen: dict[tuple[int, int], bool] = {}

    def go(i: int, j: int) -> bool:
        key = (i, j)
        if key in seen:
            return seen[key]
        if i == len(pats):
            r = j == len(segs)
        elif pats[i] == "**":
            r = go(i + 1, j) or (j < len(segs) and go(i, j + 1))
        else:
            r = j < len(segs) and _seg_match(pats[i], segs[j]) and go(i + 1, j + 1)
        seen[key] = r
        return r

    return go(0, 0)


def load_globs(manifest: str | Path | None = None) -> Globs:
    """The manifest's prompt-bearing glob list. ``ManifestError`` if the file is absent, unreadable, not the
    ``ccg-eval-manifest-v1`` schema, or its ``promptBearingMarkdown`` block is not exactly ``{include, except}`` lists."""
    path = Path(manifest) if manifest is not None else MANIFEST
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ManifestError(f"{path}: larger than {_MAX_MANIFEST_BYTES} bytes")
        doc = yaml.safe_load(path.read_bytes())
    except FileNotFoundError as e:
        raise ManifestError(f"eval manifest absent: {path}") from e
    except (OSError, yaml.YAMLError) as e:
        raise ManifestError(f"eval manifest unreadable: {path}: {e.__class__.__name__}") from e
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ManifestError(f"{path}: not a {SCHEMA} manifest")
    block = doc.get("promptBearingMarkdown")
    if not isinstance(block, dict) or set(block) != {"include", "except"}:
        raise ManifestError("promptBearingMarkdown must be a mapping with exactly the keys include and except")
    return Globs(include=_check_patterns("include", block["include"], non_empty=True),
                 exclude=_check_patterns("except", block["except"], non_empty=False))
