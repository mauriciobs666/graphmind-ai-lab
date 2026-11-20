"""What every source kind shares: path-derived identity, hashing, the header fields."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..discover import component_of, slug_of
from .header import Header, parse_header

# Bumping forces `sync --all` to re-parse every source (graph note section 2.2, G6).
PARSER_VERSION = 1


def decode(raw: bytes) -> str:
    """UTF-8 with replacement, the decoding `wi`'s own header reader uses."""
    return raw.decode("utf-8", "replace")


@dataclass
class SourceDoc:
    path: str
    kind: str
    sha256: str
    size: int
    header: Header | None
    warnings: list[str] = field(default_factory=list)

    @property
    def component(self) -> str:
        return component_of(self.path)

    @property
    def slug(self) -> str:
        return slug_of(self.path)

    def source_props(self) -> dict:
        """The `:Source` properties a parse determines (graph note section 2.2).

        `warnings` is stored for every kind, never absent, so a clean parse differs from no parse.

        Absent-vs-empty follows the note: `docStatus`/`owner`/`tracks`/`itemId` are left out
        when not recorded, `aliases` is `[]` for a header block without `Aliases:` and left
        out when the file has no header block.
        """
        h = self.header
        props: dict = {
            "path": self.path,
            "kind": self.kind,
            "component": self.component,
            "slug": self.slug,
            "sha256": self.sha256,
            "bytes": self.size,
            "parserVersion": PARSER_VERSION,
            "warnings": list(self.warnings),
        }
        if h is not None:
            for name, value in (
                ("docStatus", h.doc_status), ("owner", h.owner),
                ("tracks", h.tracks), ("itemId", h.item_id),
            ):
                if value is not None:
                    props[name] = value
            props["aliases"] = list(h.aliases)
        return props


def base_fields(path: str, kind: str, raw: bytes) -> tuple[str, dict]:
    """`(text, kwargs for SourceDoc)` for raw file bytes."""
    text = decode(raw)
    header = parse_header(text)
    warnings = list(header.warnings) if header else []
    return text, {
        "path": path, "kind": kind, "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw), "header": header, "warnings": warnings,
    }
