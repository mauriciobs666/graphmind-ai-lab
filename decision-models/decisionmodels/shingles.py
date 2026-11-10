"""8-token shingles over a generated item's rendered evidence (note
`docs/plans/typed-decision-models-ml.md` §Q1.3 "Shingle check" and §Q1.4; plan2
`docs/plans/typed-decision-models2.md` §3.5; plan4 §3.6 "Cells, then splits").

Pure functions, standard library only.

- **Tokens** (§Q1.3): NFKC-normalize and lowercase, replace every character that is not a Unicode
  letter or digit with a space, split on whitespace. A **shingle** is 8 consecutive tokens
  (`SHINGLE_TOKENS`); a text of fewer tokens has none.
- **Evidence** (`evidence_text`) is the item's rendered evidence fields' text only, per point.
  Condition sentences (guard's `condition`) and the agent's name are not evidence, so a sentence
  every item of a cell shares by design never matches. Speaker names are not text either. A
  guard's `understanding` contributes its string values in sorted key order (`known`, `missing`,
  `request`; never its key names, never the item's own order). Fields are joined by a newline, so a
  shingle may straddle two fields.
- **Side-check hashes** (§Q1.4): `decisionmodels/` never reads the 85 side-check items. A
  human-invoked script (`scripts/import_side_check_shingles.py`) writes the sha256 of each of their
  shingles, one hex digest per line under `#` header lines, to
  `data/guard-judge/side-check-shingles.txt`; `load_side_check_hashes` reads that file, and
  `shingle_hash` is the one hash function both sides use. A hash cannot serve as a seed or an
  example. A paraphrase that evades an exact-text check still shares an 8-token span.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SHINGLE_TOKENS = 8
SIDE_CHECK_FILE = Path(__file__).resolve().parents[1] / "data" / "guard-judge" / (
    "side-check-shingles.txt")


def tokens(text: str) -> list[str]:
    """Note §Q1.3's tokenization (module docstring)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c if c.isalpha() or c.isdigit() else " " for c in text).split()


def shingles(text: str) -> set[tuple[str, ...]]:
    """Every run of `SHINGLE_TOKENS` consecutive tokens of `text`."""
    words = tokens(text)
    return {tuple(words[i:i + SHINGLE_TOKENS]) for i in range(len(words) - SHINGLE_TOKENS + 1)}


def shingle_hash(shingle: Iterable[str]) -> str:
    """sha256 over the UTF-8 of the shingle's tokens joined by one space: the form the side-check
    file holds."""
    return hashlib.sha256(" ".join(shingle).encode("utf-8")).hexdigest()


def shingle_hashes(text: str) -> set[str]:
    return {shingle_hash(s) for s in shingles(text)}


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value, mappings in key order, never a key."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for key in sorted(value) for text in _strings(value[key])]
    if isinstance(value, (list, tuple)):
        return [text for entry in value for text in _strings(entry)]
    return []


def evidence_text(role: str, item: Mapping[str, Any]) -> str:
    """The text of an item's rendered evidence fields (module docstring), joined by a newline.
    Guard: `understanding` and each turn's `text`; join-in: the prior turns' `text` and the new
    message's; message type: the prior turn's `text` and the message `text`."""
    if role == "guard-judge":
        parts = _strings(item.get("understanding") or {}) + [
            turn["text"] for turn in item.get("turns") or []]
    elif role == "join-in-judge":
        parts = [turn["text"] for turn in item.get("priorTurns") or []]
        parts.append(item["newMessage"]["text"])
    elif role == "message-type-classifier":
        prior = item.get("priorTurn")
        parts = ([prior["text"]] if prior else []) + [item["text"]]
    else:
        raise ValueError(f"no evidence fields are defined for role {role!r}")
    return "\n".join(parts)


def load_side_check_hashes(path: Path = SIDE_CHECK_FILE) -> frozenset[str]:
    """The hashes in a side-check file: its non-blank lines that do not start with `#`. A line
    that is not a 64-digit hex digest raises `ValueError`, so a damaged file cannot pass as an
    empty one; a missing file raises `FileNotFoundError`."""
    found: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
            raise ValueError(f"{path} line {number} is not a sha256 hex digest")
        found.add(line)
    return frozenset(found)


def hits_side_check(text: str, side_check: frozenset[str] | set[str]) -> bool:
    """Whether any shingle of `text` is one of the side-check hashes."""
    return not shingle_hashes(text).isdisjoint(side_check)
