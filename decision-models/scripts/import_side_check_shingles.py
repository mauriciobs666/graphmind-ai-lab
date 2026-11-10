"""Write the side-check items' 8-token shingle hashes (plan2 `docs/plans/typed-decision-models2.md`
§3.5; note `docs/plans/typed-decision-models-ml.md` §Q1.4). Human-invoked, never by the pipeline.

    .venv/bin/python scripts/import_side_check_shingles.py [--items PATH] [--out PATH]

Reads the 85 human-curated guard items (`model-bench/packs/guard-judge-understanding/items.jsonl`)
and writes **only** `shingles.shingle_hash` digests of their evidence shingles, one per line,
sorted, to `data/guard-judge/side-check-shingles.txt` under two `#` header lines: the source file's
sha256 and the shingle length. No token, no item text and no item id is written, so the output
cannot serve as a seed or an example. This is the one file under `decision-models/` allowed to name
the side-check pack (`tests/test_fr8_isolation.py`'s exemption map); `decisionmodels/` never reads
it, and reads only the hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decisionmodels import shingles  # noqa: E402

DEFAULT_ITEMS = (ROOT.parent / "model-bench" / "packs" / "guard-judge-understanding"
                 / "items.jsonl")


def build(items: bytes) -> str:
    """The side-check file's text for the bytes of an `items.jsonl`."""
    hashes: set[str] = set()
    for line in items.decode("utf-8").splitlines():
        if line.strip():
            hashes |= shingles.shingle_hashes(
                shingles.evidence_text("guard-judge", json.loads(line)))
    header = [f"# source-sha256: {hashlib.sha256(items).hexdigest()}",
              f"# shingle-tokens: {shingles.SHINGLE_TOKENS}"]
    return "\n".join(header + sorted(hashes)) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out", type=Path, default=shingles.SIDE_CHECK_FILE)
    args = parser.parse_args(argv)
    if not args.items.is_file():
        print(f"refused: {args.items} is not a file", file=sys.stderr)
        return 1
    text = build(args.items.read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {text.count(chr(10)) - 2} shingle hashes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
