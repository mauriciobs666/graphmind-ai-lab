"""Pure text chunking for document ingestion (K-050 M5 Stage 1).

No I/O — `split_into_chunks` is the only public entry point, trivially
unit-testable. Boundary rule, in priority order (plan §3.2):

  1. prefer paragraph breaks (``\\n\\n``);
  2. if a paragraph still exceeds `size`, split on sentence boundaries
     (``". "``, ``"! "``, ``"? "``, or a bare ``"\\n"``);
  3. if a "sentence" still exceeds `size`, hard-cut at `size` characters,
     carrying `overlap` characters forward regardless of boundary.

See `docs/plans/document-ingestion.md` §3.2 for the design rationale.
"""

from __future__ import annotations

_PARAGRAPH_DELIMS: tuple[str, ...] = ("\n\n",)
_SENTENCE_DELIMS: tuple[str, ...] = (". ", "! ", "? ", "\n")


def split_into_chunks(text: str, *, size: int = 1000, overlap: int = 150) -> list[str]:
    """Split `text` into chunks of at most ~`size` characters each.

    Each chunk after the first is prefixed with the trailing `overlap`
    characters of the previous chunk, so a fact split across a chunk
    boundary is still extractable from at least one chunk. Empty or
    whitespace-only input yields ``[]`` — chunking is not the layer that
    rejects that (the service validates non-empty input upstream).
    """
    if not text or not text.strip():
        return []
    pieces: list[str] = []
    for paragraph in _split_keep_delim(text, _PARAGRAPH_DELIMS):
        pieces.extend(_split_oversized(paragraph, size))
    return _pack_with_overlap(pieces, size=size, overlap=overlap)


def _split_keep_delim(text: str, delims: tuple[str, ...]) -> list[str]:
    """Split `text` on any of `delims`, keeping each delimiter attached to the
    piece that precedes it, so ``"".join(_split_keep_delim(t, d)) == t``
    always holds. The longest delimiter is tried first at each position.
    """
    ordered = sorted(delims, key=len, reverse=True)
    pieces: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        hit = next((d for d in ordered if d and text.startswith(d, i)), None)
        if hit:
            i += len(hit)
            pieces.append(text[start:i])
            start = i
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return pieces


def _split_oversized(unit: str, size: int) -> list[str]:
    """Boundary-rule steps 2/3: shrink `unit` below `size` if it isn't already.

    Tries sentence boundaries first; a piece with no sentence boundary at all
    (a single run-on "sentence") falls through to a hard cut. Recurses once
    per sentence so a run of several sentences where only one is oversized
    only hard-cuts that one.
    """
    if len(unit) <= size:
        return [unit]
    sentences = _split_keep_delim(unit, _SENTENCE_DELIMS)
    if len(sentences) <= 1:
        return _hard_cut(unit, size)
    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_oversized(sentence, size))
    return pieces


def _hard_cut(text: str, size: int) -> list[str]:
    """Boundary-rule step 3: cut every `size` characters, no boundary awareness."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _pack_with_overlap(pieces: list[str], *, size: int, overlap: int) -> list[str]:
    """Greedily pack atomic `pieces` (each already <= `size`) into chunks.

    A finished chunk's trailing `overlap` characters are carried into the
    start of the next one — the same mechanism whether the boundary being
    crossed is a natural paragraph/sentence break or a hard cut (the plan's
    "regardless of boundary" clause for step 3 is just this generic step
    applying there too, not a separate code path).
    """
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) <= size:
            current += piece
        else:
            chunks.append(current)
            carry = current[-overlap:] if overlap > 0 else ""
            current = carry + piece
    if current:
        chunks.append(current)
    return chunks
