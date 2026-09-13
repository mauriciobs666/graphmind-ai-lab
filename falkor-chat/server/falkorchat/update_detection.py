"""Document update-detection — pure functions only (K-050 M5 follow-on,
document-ingestion2 Stage D, FR-2/AC-1/AC-2).

Mirrors `fusion.py`'s shape: no I/O, no Cypher, no repository access — every
function here takes plain Python values and returns plain Python values. The
graph-facing half (the atomic auto-supersede write, the LSH-band-anchored
candidate shortlist, the find-or-reopen suggestion write) lives in
`repository.py`; the orchestration that calls both halves together lives in
`background.py` (`_safe_detect_update`), exactly the same split `fusion.py`
already has with `ingestion.IngestionPipeline`/`repository.py`.

Per the ML note (`docs/plans/document-ingestion2-ml.md` §3): **no embeddings,
no LLM, no calibrated numeric threshold anywhere in v1.**

* Auto tier (FR-2/AC-1) — exact content identity: `content_hash(normalize_text(text))`
  equality against an existing `currentVersion` document. `normalize_text` is
  the **one** shared normalizer for both the hash and the shingling below (the
  ML note's own instruction not to write two independently-drifting
  normalizers — mirrors `extraction.normalize_name`'s "one shared helper"
  precedent).
* Suggested tier (FR-2/AC-2) — shingled-Jaccard content overlap
  (`shingles`/`jaccard`) over a cheaply-narrowed candidate shortlist. The
  shortlist itself is narrowed via an LSH/MinHash-banding fingerprint
  (`minhash_signature`/`lsh_bands`) computed off the same shingle set
  `jaccard` uses — `graph-dba`'s redesign of the candidate-generation index
  (plan §0/§3.4/§7), replacing a RAM-heavy raw full-text index on
  `Document.text`. The stored `SUPERSEDES.confidence` is always the precise
  raw Jaccard ratio (never an LSH estimate) — the bands only ever narrow
  *candidates*.
"""

from __future__ import annotations

import hashlib
import random
import re

_WHITESPACE_RE = re.compile(r"\s+")

# MinHash universal-hash-family coefficients, generated once at import time
# from a FIXED seed — deterministic and PROCESS-STABLE (unlike Python's
# built-in `hash()`, which is salted per-process via `PYTHONHASHSEED` and
# would make two server processes derive different `lshBand<i>` values for
# byte-identical shingle sets). `k`/`b` (the caller's tunable knobs, plan §4
# Stage D) only ever select a PREFIX of this fixed sequence, so results stay
# deterministic across the codebase's default `k=32`/`b=8` and any smaller
# override a future caller might pass.
_MINHASH_SEED = 1337
# A prime comfortably larger than any 64-bit token hash, for the
# `(a*x + b) mod p` universal hash family.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_MINHASH_K = 256


def _hash_coefficients(k: int) -> list[tuple[int, int]]:
    rng = random.Random(_MINHASH_SEED)
    return [
        (rng.randint(1, _MERSENNE_PRIME - 1), rng.randint(0, _MERSENNE_PRIME - 1))
        for _ in range(max(k, _MAX_MINHASH_K))
    ][:k]


def _stable_token_hash(token: str) -> int:
    """A 64-bit hash of `token` that is stable across processes/platforms —
    `hashlib.md5` (not Python's salted built-in `hash()`) so two server
    processes derive the identical MinHash signature for the same shingle,
    which `lsh_bands`'s whole candidate-generation index depends on. Not a
    security use of MD5 — collision resistance is irrelevant here, only
    speed and cross-process stability."""
    digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not a security use
    return int.from_bytes(digest[:8], "big")


def normalize_text(text: str) -> str:
    """Case-fold + whitespace-collapse — the **one** shared normalizer for
    both `content_hash` (auto tier) and `shingles` (suggested tier), per the
    ML note's own instruction (§3.2) not to write two independently-drifting
    normalizers. Collapses every run of whitespace (including newlines/tabs)
    to a single space, strips leading/trailing whitespace, then case-folds
    (broader than `.lower()` for non-ASCII text, same rationale
    `extraction.normalize_name` already applies to entity names).
    """
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def content_hash(normalized_text: str) -> str:
    """SHA-256 hex digest of already-normalized text — the auto tier's exact-
    identity criterion (`Document.textNormalizedHash`, ML note §3.1). A
    definitional identity check, not a score: two documents whose
    `content_hash(normalize_text(text))` values match are, for any real
    corpus, the same content, full stop.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def shingles(normalized_text: str, n: int = 5) -> set[str]:
    """`n`-word shingles (default 5, ML note §3.2) of already-normalized text.

    Empty/whitespace-only input yields an empty set. A document shorter than
    `n` words yields exactly one shingle (the whole document) rather than an
    empty set — a short-but-nonempty document should still be comparable via
    Jaccard, not silently invisible to the suggested tier.
    """
    tokens = normalized_text.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """`|A ∩ B| / |A ∪ B|` — the suggested tier's raw content-overlap ratio
    (ML note §3.2), stored verbatim as `SUPERSEDES.confidence` — "a content-
    overlap measurement between 0 and 1, not a calibrated probability that
    this is the same document" (ML note §5).

    Two empty shingle sets (both documents reduced to nothing, e.g.
    whitespace-only text) return `0.0`, not `1.0` — there is no shared
    content to measure an overlap *of*; this deliberately does not lean on
    Jaccard to express identity (that is `content_hash` equality's job, the
    auto tier, not this function's).
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def minhash_signature(shingles: set[str], k: int = 32) -> list[int]:
    """A standard MinHash signature of length `k` over `shingles` — the same
    shingle set `jaccard` itself compares (no second document representation,
    plan §4 Stage D). `k` is an implementer-tunable RAM/recall trade-off
    (mirrors `MAX_DOCUMENT_CHARS`'s "implementer-tunable, not load-bearing"
    posture), not load-bearing at any specific value.

    An empty shingle set (an empty/whitespace-only document) yields a
    signature of `k` zeros — a degenerate but well-defined value, so
    `lsh_bands` never has to special-case it.
    """
    if not shingles:
        return [0] * k
    token_hashes = [_stable_token_hash(s) for s in shingles]
    coefficients = _hash_coefficients(k)
    return [
        min((a * h + b) % _MERSENNE_PRIME for h in token_hashes)
        for a, b in coefficients
    ]


def lsh_bands(signature: list[int], b: int = 8) -> list[str]:
    """Bands a MinHash `signature` into `b` short fingerprint strings, one per
    `Document.lshBand<i>` property (plan §4 Stage D) — the standard LSH-
    banding technique for near-duplicate candidate generation: two documents
    whose MinHash signatures agree on an entire band collide on that band's
    fingerprint, so an equality lookup on any single `lshBand<i>` (an
    index-anchored `OR` across all `b`, `repository.find_update_shortlist`)
    surfaces likely-similar candidates cheaply, without ever comparing raw
    text. `b` is an implementer-tunable knob, same posture as `k` above; the
    signature is split as evenly as possible across `b` bands (the last band
    absorbs any remainder when `len(signature)` doesn't divide evenly).

    An empty/degenerate signature (all zeros, from an empty shingle set)
    still yields `b` well-defined fingerprint strings — they will simply
    collide with any other empty document's bands, which is the correct
    behavior (two empty documents are, trivially, near-duplicates of nothing
    in particular but of each other under this signal).
    """
    if not signature:
        return ["" for _ in range(b)]
    rows_per_band = max(1, len(signature) // b)
    bands: list[str] = []
    for i in range(b):
        start = i * rows_per_band
        end = start + rows_per_band if i < b - 1 else len(signature)
        chunk = signature[start:end]
        band_bytes = ",".join(str(v) for v in chunk).encode("utf-8")
        bands.append(hashlib.md5(band_bytes).hexdigest()[:16])  # noqa: S324 - not a security use
    return bands
