"""Move `openPointsS27` from wherever the custodians nested it in `adjudication/<lang>.noise.json` to the
top level (v1.12 §5.4 step 7: a required top-level key; `validate_final.py` NOISE_KEYS).

    python hoist_open_points.py [--adjudication DIR] [lang ...]

Deterministic and idempotent: the nested key is found by name inside `detail` (exactly one occurrence, else
refuse), removed from its parent, and inserted just before `detail`; nothing else changes (asserted by
comparing the document with the key removed from both sides). A file already carrying it at top level with no
nested copy is left untouched. Files must be in the custodians' format (indent=1, ASCII-escaped, no trailing
newline), which is preserved; any other formatting is refused unchanged rather than silently reformatted.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

KEY = "openPointsS27"
LANGS = ("python", "javascript", "go", "java", "csharp", "cpp", "kotlin")
HERE = Path(__file__).resolve().parent


def _find(node, path, found):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == KEY:
                found.append((path, node))
            _find(v, path + (k,), found)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _find(v, path + (i,), found)


def _without(doc, found):
    doc = copy.deepcopy(doc)
    doc.pop(KEY, None)
    nested = []
    _find(doc.get("detail"), ("detail",), nested)
    for _, parent in nested:
        parent.pop(KEY, None)
    return doc


def hoist(noise: dict):
    """Return (new_noise, changed). Raises ValueError when the key is absent, duplicated or ambiguous."""
    nested: list = []
    _find(noise.get("detail"), ("detail",), nested)
    if KEY in noise:
        if nested:
            raise ValueError(f"{KEY} at top level and nested at {[p for p, _ in nested]}")
        return noise, False
    if len(nested) != 1:
        raise ValueError(f"expected exactly one nested {KEY}, found {len(nested)}")
    _, parent = nested[0]
    value = parent[KEY]
    out = {}
    for k, v in noise.items():
        if k == "detail":
            out[KEY] = value
        out[k] = v
    del parent[KEY]
    return out, True


def _format(doc) -> str:
    return json.dumps(doc, indent=1)


def hoist_file(path) -> bool:
    path = Path(path)
    text = path.read_text()
    before = json.loads(text)
    if text != _format(before):
        raise ValueError(f"{path.name}: unrecognised formatting (expected indent=1, ASCII, no trailing newline)")
    original = copy.deepcopy(before)
    after, changed = hoist(before)
    if not changed:
        return False
    # nothing but the move: removing the key from both documents (top level and nested) must give equal documents
    if _without(after, None) != _without(original, None):
        raise AssertionError(f"{path.name}: hoist changed more than {KEY}")
    value_before = []
    _find(original["detail"], ("detail",), value_before)
    assert after[KEY] == value_before[0][1][KEY]
    path.write_text(_format(after))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--adjudication", default=str(HERE / "adjudication"))
    a = ap.parse_args(argv)
    for lang in a.langs or LANGS:
        p = Path(a.adjudication) / f"{lang}.noise.json"
        print(f"{lang}: {'hoisted' if hoist_file(p) else 'already top-level, untouched'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
