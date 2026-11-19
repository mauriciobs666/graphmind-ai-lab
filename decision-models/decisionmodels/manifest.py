"""The fit manifest: the bytes a fit writes and a candidate declares
(`docs/plans/typed-decision-models4.md` §3.11; `docs/plans/typed-decision-models.md` v1.3 §3.4;
model-bench `decisions.load_fit_manifest`).

The file is UTF-8 text: one `input_hash` per line, each a lowercase-hex sha256, each line ended by
a single LF (the last included; no CR, no BOM), lines strictly ascending. The empty file is the
manifest of zero inputs. A candidate declares `fitManifest.sha256` = sha256 over the raw bytes and
`fitManifest.size` = the number of entries, never the byte length.
`tests/fixtures/decision-contract/fit-manifest.jsonl` pins this, byte-copied from model-bench; this
is the format implemented again (FR-8 forbids importing model-bench), writer side included, with
the reader's checks and order.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestRefused(ValueError):
    """A manifest that cannot carry the leakage check; `rule` names the broken rule."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


def render_manifest(input_hashes: Iterable[str]) -> bytes:
    """The manifest bytes of these `input_hash` values: sorted, one per line, LF-terminated.
    A repeated hash collapses to one entry; a value not a lowercase-hex sha256 is refused."""
    entries = list(input_hashes)
    for entry in entries:
        if not isinstance(entry, str) or not _SHA256_RE.fullmatch(entry):
            raise ManifestRefused("malformed", f"not a lowercase-hex sha256: {entry!r}")
    entries = sorted(set(entries))
    return "".join(f"{entry}\n" for entry in entries).encode("utf-8")


def parse_manifest(raw: bytes) -> tuple[str, ...]:
    """The entries of manifest bytes, refusing anything `render_manifest` could not have written."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestRefused("not-utf8", f"not UTF-8: {exc}") from exc
    if text and not text.endswith("\n"):
        raise ManifestRefused("no-final-newline", "the last line has no terminating newline")
    lines = text.split("\n")[:-1] if text else []
    for number, line in enumerate(lines, start=1):
        if _SHA256_RE.fullmatch(line):
            continue
        if line.endswith("\r"):
            rule = "crlf"
        elif _SHA256_RE.fullmatch(line.lower()):
            rule = "uppercase"
        else:
            rule = "malformed"
        raise ManifestRefused(rule, f"line {number} is not a lowercase-hex sha256: {line!r}")
    for a, b in zip(lines, lines[1:]):
        if a == b:
            raise ManifestRefused("repeated", "entries are not strictly ascending (repeated)")
        if a > b:
            raise ManifestRefused("unsorted", "entries are not strictly ascending (unsorted)")
    return tuple(lines)


def declare_manifest(raw: bytes) -> dict[str, object]:
    """The candidate's `fitManifest` declaration for these bytes: `sha256` over the raw bytes and
    `size` as the entry count. Refuses bytes `parse_manifest` refuses."""
    entries = parse_manifest(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(entries)}


def verify_manifest(raw: bytes, sha256: object, size: object) -> tuple[str, ...]:
    """The entries of `raw` if it matches the declaration, else `ManifestRefused`. The sha256 is
    checked first, then the format, then the count; each independently of the others. A bool
    `size` is refused, deliberately stricter than the bench's reader, which accepts `True` as 1."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ManifestRefused(
            "wrong-sha256", f"sha256 {digest} is not the declared fitManifest.sha256 {sha256}"
        )
    entries = parse_manifest(raw)
    if len(entries) != size or isinstance(size, bool):
        raise ManifestRefused(
            "wrong-size", f"{len(entries)} entries, not the declared fitManifest.size {size!r}"
        )
    return entries
